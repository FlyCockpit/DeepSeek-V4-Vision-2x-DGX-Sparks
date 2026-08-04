"""DSV4-Flash-0731 + DeepEncoderV2 vision adapter, as a vLLM model wrapper.

Why a wrapper and not a core patch
----------------------------------
DSV4's MoE is a HashRouter: expert selection is `tid2eid[input_ids]`, keyed on
literal TOKEN IDS, not hidden state. vLLM's multimodal path normally hands the
model `inputs_embeds` and sets `input_ids = None`, which would break routing --
vllm/models/deepseek_v4/nvidia/model.py raises
"DeepSeek V4 hash MoE routing requires input_ids." outright.

vLLM already has the escape hatch: `requires_raw_input_tokens` (a ClassVar on
SupportsMultiModal). When True, gpu_model_runner._prepare_mm_inputs passes BOTH
`input_ids` and `inputs_embeds`, and DSV4's forward threads `input_ids` into every
decoder layer independently of `inputs_embeds`. So the splice needs no core patch.

Layout contract
---------------
    n_img_tokens = n_views * 256 + 1
    rows [0 : n_views*256] <- projector(tower(pixels))
    row  [n_views*256]     <- adapter.view_seperator, a LEARNED nn.Parameter(4096)

The separator row is NOT a feature. Filling all rows with features raises no error
and still yields fluent text -- it just feeds a layout the model never saw.

Ordering is GLOBAL view first, then tiles, then the separator (global_view_pos
"head"). Tile count comes from the checkpoint (`config.tiles`); it is not
hardcoded to a fixed multi-view layout.
"""

import os
import sys
from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

# --- frozen interface constants ---------------------------------------------
IMAGE_TOKEN_ID = 129279
IMAGE_TOKEN_STR = "<｜image｜>"
ENCODER_DIM = 896
HIDDEN_SIZE = 4096
IMAGE_SIZE = 1024
TOKENS_PER_VIEW = 256

MIN_SCALE = 1.5          # tile only when max(w,h) >= 1.5 * 1024 = 1536

TOWER_PATH = os.environ.get("DSV4_VISION_TOWER")
ADAPTER_PATH = os.environ.get("DSV4_VISION_ADAPTER")

_TILES_CACHE: list = []


def checkpoint_tiles() -> int:
    """`tiles` the served checkpoint was TRAINED with, read from the checkpoint.

    Read `tiles` from the checkpoint config; do not hardcode. Older tiles=0
    adapters stay at a fixed 257 tokens per image.
    """
    if _TILES_CACHE:
        return _TILES_CACHE[0]
    override = os.environ.get("DSV4_VISION_TILES")
    if override is not None:
        _TILES_CACHE.append(int(override))
        return _TILES_CACHE[0]
    tiles = 0
    if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
        try:
            ck = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
            # The key is `config`. It is NOT `cfg` -- reading `cfg` returns None,
            # falls through to 0, and would serve a TILED adapter with a 257-token
            # layout: no error, just a layout the model never trained on.
            meta = ck.get("config")
            if meta is None:
                meta = ck.get("cfg")
            if meta is None:
                # Older adapters may omit config entirely → treat as tiles=0.
                print("[dsv4-vision] WARNING: no `config` in "
                      f"{ADAPTER_PATH}; assuming tiles=0 (257 tokens/image). "
                      "Set DSV4_VISION_TILES to override.", file=sys.stderr)
            else:
                tiles = int(meta.get("tiles", 0) if isinstance(meta, dict)
                            else getattr(meta, "tiles", 0) or 0)
                print(f"[dsv4-vision] checkpoint config.tiles={tiles} "
                      f"({ADAPTER_PATH})", file=sys.stderr)
        except Exception as exc:  # never fail startup on metadata
            print(f"[dsv4-vision] WARNING: could not read config from "
                  f"{ADAPTER_PATH}: {exc}; assuming tiles=0", file=sys.stderr)
            tiles = 0
    _TILES_CACHE.append(tiles)
    return tiles


def best_grid(w: int, h: int, max_tiles: int) -> tuple[int, int]:
    """cols/rows closest to the aspect ratio; ties go to MORE tiles.
    A square grid inherits the squeeze a square resize applies to 16:9, so the
    grid has to be aspect-aware."""
    ar, best_key, best = w / h, None, (1, 1)
    for cols in range(1, max_tiles + 1):
        for rows in range(1, max_tiles + 1):
            n = cols * rows
            if n > max_tiles or n == 1:
                continue
            key = (abs(cols / rows - ar), -n)
            if best_key is None or key < best_key:
                best_key, best = key, (cols, rows)
    return best


def crop_boxes(w: int, h: int, tiles: int) -> list[tuple[int, int, int, int]]:
    """Crop boxes in ROW-MAJOR order (rows outer, cols inner), or [] for
    global-view-only. INTEGER division is load-bearing: boxes can differ by a
    pixel and must match the reference exactly."""
    if tiles > 1 and max(w, h) >= MIN_SCALE * IMAGE_SIZE:
        cols, rows = best_grid(w, h, tiles * tiles)
        return [
            (c * w // cols, r * h // rows, (c + 1) * w // cols, (r + 1) * h // rows)
            for r in range(rows)
            for c in range(cols)
        ]
    return []


def num_views_for(w: int, h: int, tiles: int) -> int:
    return 1 + len(crop_boxes(w, h, tiles))          # global view is always present


def n_img_tokens_for(w: int, h: int, tiles: int) -> int:
    return num_views_for(w, h, tiles) * TOKENS_PER_VIEW + 1   # +1 separator


def render_views(image, tiles: int) -> torch.Tensor:
    """(n_views, 3, 1024, 1024). GLOBAL view FIRST, then crops row-major.
    Every view -- crops included -- is resized to a 1024 SQUARE, not to its own
    aspect ratio."""
    w, h = image.size
    views = [image] + [image.crop(b) for b in crop_boxes(w, h, tiles)]
    return torch.stack([preprocess_image(v) for v in views])


def preprocess_image(image) -> torch.Tensor:
    """PIL image -> (3, 1024, 1024). Must match training exactly:
    resize 1024x1024 BICUBIC, RGB, x/255, then (x-0.5)/0.5."""
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = torch.frombuffer(image.tobytes(), dtype=torch.uint8).clone()
    arr = arr.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float().div_(255.0)
    return arr.sub_(0.5).div_(0.5)


class Adapter(nn.Module):
    """The trained projector: Linear(896->4096) -> GELU -> Linear(4096->4096),
    plus a learned view_seperator parameter.
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ENCODER_DIM, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        )
        self.view_seperator = nn.Parameter(torch.zeros(HIDDEN_SIZE))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)


class DSV4VisionProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        # PER IMAGE, not a constant: with aspect-aware tiling the count depends
        # on this image's own dimensions (257 / 769 / 1281).
        return n_img_tokens_for(image_width, image_height, checkpoint_tiles())

    def get_image_size_with_most_features(self) -> ImageSize:
        # Profile against the worst case so the encoder cache budget is sized for
        # it: near-square above the threshold takes a 2x2 grid = 5 views.
        if checkpoint_tiles() > 1:
            return ImageSize(width=2560, height=2168)
        return ImageSize(width=IMAGE_SIZE, height=IMAGE_SIZE)


class DSV4VisionDummyInputsBuilder(BaseDummyInputsBuilder[DSV4VisionProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN_STR * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
            )
        }


class DSV4VisionMultiModalProcessor(
    BaseMultiModalProcessor[DSV4VisionProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        out = dict(enc)

        images = mm_data.get("images", mm_data.get("image"))
        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            if images:
                tiles = checkpoint_tiles()
                per_image = [render_views(im, tiles) for im in images]
                # Flattened across images; num_views lets vLLM slice it back apart,
                # since the view count now varies per image.
                out["pixel_values"] = torch.cat(per_image, dim=0)
                out["num_views"] = torch.tensor([p.shape[0] for p in per_image],
                                                dtype=torch.long)
        return BatchFeature(out)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_views = hf_inputs.get("num_views", torch.empty((0,), dtype=torch.long))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_views),
            num_views=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        images = mm_items.get_items("image", ImageProcessorItems)

        def replacement(item_idx: int):
            size = images.get_image_size(item_idx)
            n = self.info.get_num_image_tokens(
                image_width=size.width, image_height=size.height)
            return [IMAGE_TOKEN_ID] * n

        return [
            PromptReplacement(
                modality="image",
                target=[IMAGE_TOKEN_ID],
                replacement=replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    DSV4VisionMultiModalProcessor,
    info=DSV4VisionProcessingInfo,
    dummy_inputs=DSV4VisionDummyInputsBuilder,
)
class DeepseekV4VisionForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    # THE CRUX: keep input_ids alongside inputs_embeds so the HashRouter works.
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_TOKEN_STR
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        # The backbone is plain DSV4: hand it a config that names the text
        # architecture so vLLM builds DeepseekV4ForCausalLM, not this wrapper.
        import copy

        text_config = copy.deepcopy(config)
        text_config.architectures = ["DeepseekV4ForCausalLM"]

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        dtype = vllm_config.model_config.dtype
        self._build_vision(dtype)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _build_vision(self, dtype: torch.dtype) -> None:
        """Frozen SAM+Qwen2 tower and the trained BF16 adapter.

        Both are plain torch.nn modules, so the backbone's fp8 quant_config never
        touches them -- the adapter is kept BF16.
        """
        from dsv4_vision_vllm.deepencoderv2 import (
            build_qwen2_decoder_as_encoder,
            build_sam_vit_b,
        )

        self.sam_model = build_sam_vit_b()
        self.qwen2_model = build_qwen2_decoder_as_encoder()
        self.adapter = Adapter()

        if TOWER_PATH:
            from safetensors.torch import load_file

            sd = load_file(TOWER_PATH)
            remap = {
                k[len("model."):]: v
                for k, v in sd.items()
                if k.startswith(("model.sam_model.", "model.qwen2_model."))
            }
            missing, unexpected = self.load_state_dict(remap, strict=False)
            unexpected = [u for u in unexpected if u.startswith(("sam_model.", "qwen2_model."))]
            if unexpected:
                raise RuntimeError(f"tower unexpected keys: {unexpected[:3]}")

        if ADAPTER_PATH:
            ckpt = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
            state = ckpt.get("adapter", ckpt)
            missing, unexpected = self.adapter.load_state_dict(state, strict=True)

        for p in self.sam_model.parameters():
            p.requires_grad_(False)
        for p in self.qwen2_model.parameters():
            p.requires_grad_(False)

        self.sam_model = self.sam_model.to(dtype=dtype).eval()
        self.qwen2_model = self.qwen2_model.to(dtype=dtype).eval()
        self.adapter = self.adapter.to(dtype=dtype).eval()

    @torch.no_grad()
    def _encode_one(self, views: torch.Tensor) -> torch.Tensor:
        """All views of ONE image -> (n_views*256 + 1, 4096).

        Views arrive already ordered global-first then crops row-major, so a
        plain reshape preserves that order. The separator is appended LAST and
        is a learned parameter, not a feature.
        """
        device = next(self.language_model.parameters()).device
        dtype = next(self.adapter.parameters()).dtype
        px = views.to(device=device, dtype=dtype)
        if px.dim() == 3:
            px = px.unsqueeze(0)

        feats = self.qwen2_model(self.sam_model(px))        # (n_views, 256, 896)
        proj = self.adapter(feats)                           # (n_views, 256, 4096)
        sep = self.adapter.view_seperator[None, :].to(proj.dtype)
        return torch.cat([proj.reshape(-1, HIDDEN_SIZE), sep], dim=0)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        pixel_values = kwargs.pop("pixel_values", None)
        num_views = kwargs.pop("num_views", None)
        if pixel_values is None:
            return None

        # flat_from_sizes usually hands back one entry per image already; fall
        # back to splitting a flat tensor by num_views when it does not.
        if isinstance(pixel_values, (list, tuple)):
            groups = [p if p.dim() == 4 else p.unsqueeze(0) for p in pixel_values]
        else:
            px = pixel_values
            if px.dim() == 5:
                px = px.flatten(0, 1)
            if num_views is not None:
                sizes = torch.as_tensor(num_views).flatten().tolist()
                groups = list(torch.split(px, [int(s) for s in sizes], dim=0))
            else:
                groups = [px]

        return [self._encode_one(g) for g in groups]

    def get_language_model(self) -> nn.Module:
        # SupportsMultiModal.embed_input_ids() is the merge point: it embeds the
        # text ids via get_language_model().embed_input_ids and then scatters the
        # multimodal embeddings using the is_multimodal mask. Do NOT override
        # embed_input_ids here -- its real signature is
        # (input_ids, multimodal_embeddings=None, *, is_multimodal=None), and a
        # simpler override silently breaks the scatter.
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Only the 0731 backbone comes from the checkpoint; tower and adapter are
        # loaded from their own files in _build_vision().
        loaded = self.language_model.load_weights(weights)
        return {f"language_model.{name}" for name in loaded}
