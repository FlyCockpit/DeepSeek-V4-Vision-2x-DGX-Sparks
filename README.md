# DeepSeek-V4-Flash-0731 Vision on 2× DGX Spark

Self-contained two-node DGX Spark recipe for serving
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
with a **vision encoder** (DeepEncoderV2 tower + trained projector) under vLLM
**TP=2**, **NVFP4 MLA KV**, and RoCE fabric.

This is a MiaAI-style playbook: clone, fill `.env`, download assets, build the
thin image, start worker-then-head, smoke-test.

> **Official vision encoder weights:**
> [`FlyCockpit/DeepSeek-V4-Flash-0731-vision`](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision)
>
> Final merge: **step 4800** (`adapter/merged-004800-5af0c5.pt`,
> md5 `d9b3b3bda8f790ecf7cd5a98e6fb93a5`).

---

## Result (validated on FlyCockpit Sparks, 2026-08)

| | |
|---|---|
| Backbone | `deepseek-ai/DeepSeek-V4-Flash-0731` FP8 (~167 GB) |
| Encoder | tower + adapter from HF release above |
| Layout | `tiles=2` → 257 / 769 / 1281 image tokens |
| Parallelism | TP=2, PP=1, 2 nodes, 1 GPU each |
| KV cache | `nvfp4_ds_mla` |
| Context | `max_model_len=131072` (default in this recipe) |
| Port | `8899` OpenAI-compatible `/v1` |
| Image | `dsv4-vision-vllm:0.1.1` = `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` + editable plugin |

Live checks that passed on the reference pair:

- `/v1/models` → `deepseek-v4-flash-0731-vision`, `max_model_len=131072`
- Text: `Reply with exactly: VISION_OK` → `VISION_OK`
- Image: real GUI PNG → coherent short description
- Logs: `[dsv4-vision] checkpoint config.tiles=2 (/ckpt/merged-004800-5af0c5.pt)`

### Capability honesty

Measured on the final merge (see HF card and `docs/CAPABILITIES.md`):

- **Synthetic GUI renders:** coordinate grounding works in-distribution.
- **Real screenshots / computer-use:** does **not** transfer reliably; a format
  nudge often collapses to constant `(500, 500)`.
- Do not advertise production computer-use without ScreenSpot-v2 (or better).

---

## Hardware requirements (2× path)

| item | requirement |
|---|---|
| Nodes | **2×** NVIDIA DGX Spark (GB10, aarch64, sm_121a) |
| Memory | ~121–128 GB unified per node (FP8 0731 + NVFP4 KV) |
| Link | 200G RoCE (ConnectX) between nodes; host network + `/dev/infiniband` |
| Disk | ≥200 GB free per node for HF cache + tower + adapter |
| OS | Ubuntu 24.04-class; Docker with NVIDIA runtime |

**Why 2 nodes:** the 0731 FP8 weight set is ~167 GB. A single Spark cannot hold
it for full-precision TP=1 serving. The production path is TP=2 across two
Sparks. For a 1× + 2-bit text-only option, see
[Optional: 1× Spark with Antirez 2-bit (text-only)](#optional-1-spark-with-antirez-2-bit-text-only).

---

## What gets installed

```
DeepSeek-V4-Vision-2x-DGX-Sparks/
├── README.md
├── LICENSE
├── CREDITS.md
├── .env.vision.example          # copy → .env.vision on each node role
├── docker-compose.vision.yml
├── Dockerfile                   # thin layer: base image + editable plugin
├── plugin/                      # vLLM general_plugin (bind-mounted, editable)
│   ├── pyproject.toml
│   ├── make_vision_model_dir.py
│   ├── preflight_checkpoint.py
│   ├── train_layout.jinja
│   ├── probes/
│   └── src/dsv4_vision_vllm/
├── scripts/
│   ├── download-assets.sh       # HF encoder + 0731 backbone
│   ├── build-image.sh
│   ├── start-vision.sh          # worker first, then head (Mia order)
│   ├── stop-vision.sh
│   ├── status-vision.sh
│   ├── smoke-vision.sh
│   └── preflight-adapter.sh
└── docs/
    ├── CAPABILITIES.md
    ├── NETWORK.md
    └── TROUBLESHOOTING.md
```

---

## Quick start (2×)

### 0. Prerequisites on **both** nodes

```bash
# Docker + NVIDIA container toolkit already working
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Hugging Face CLI (or use huggingface_hub in a venv)
pip install -U "huggingface_hub>=0.24"
# export HF_TOKEN=...   # only if you need private access; the release is public
```

Clone this repo to the **same absolute path** on both nodes (usernames may
differ; paths must match what you put in `.env.vision`):

```bash
git clone git@github.com:FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks.git
cd DeepSeek-V4-Vision-2x-DGX-Sparks
```

### 1. Configure fabric + paths

```bash
cp .env.vision.example .env.vision
# edit: MASTER_ADDR, head/worker fabric IPs, NCCL_*, HOME paths, WORKER_HOST
```

See [`docs/NETWORK.md`](docs/NETWORK.md). Reference fabric from the validated
pair:

| role | fabric IP | example RoCE NIC | socket ifname |
|---|---|---|---|
| head | `192.168.100.2` | `rocep1s0f1` | `enp1s0f1np1` |
| worker | `192.168.100.1` | `rocep1s0f1` | `enp1s0f1np1` |

`NCCL_IB_GID_INDEX=3` and RoCEv2 were required on the reference hardware.

### 2. Download assets (both nodes)

```bash
bash scripts/download-assets.sh
```

This pulls:

1. **Backbone:** `deepseek-ai/DeepSeek-V4-Flash-0731` into `$HF_CACHE`
2. **Encoder:** from
   [`FlyCockpit/DeepSeek-V4-Flash-0731-vision`](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision)
   - `tower/deepencoder_v2_tower.safetensors` (md5 `2d5dba62…`)
   - `adapter/latest.pt` ≡ `merged-004800-5af0c5.pt` (md5 `d9b3b3bd…`)

Then builds the symlink model dir `dsv4-0731-vision` (architecture rewrite only).

**md5-verify on both nodes before serving.** Published checksums have been
rewritten in place during provenance cleanups; a cached name is not evidence.

### 3. Build the thin image (both nodes)

```bash
bash scripts/build-image.sh
```

Base: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`  
Tag: `dsv4-vision-vllm:0.1.1`

The plugin is installed **editable** and bind-mounted, so later plugin edits are
`scp` + restart — never a rebuild.

### 4. Preflight the adapter (CPU-only, safe while live)

```bash
bash scripts/preflight-adapter.sh
# expect: PREFLIGHT_OK, tiles=2, ~20,459,520 params, healthy cosine if multi-replica meta present
```

### 5. Start (from the **head** only)

```bash
bash scripts/start-vision.sh
```

Order is fixed (same as Mia DSpark recipes):

1. Refresh model dirs on both nodes  
2. Write per-node `.env.vision`  
3. **Worker first** (`NODE_RANK=1 --headless`)  
4. **Head second** (`NODE_RANK=0`)  
5. Wait for `/v1/models` + optional chat smoke  

Cold start is on the order of **~6 minutes** (167 GB weight load).

### 6. Verify

```bash
bash scripts/status-vision.sh
bash scripts/smoke-vision.sh            # text
bash scripts/smoke-vision.sh /path/to.png  # image
```

Only announce “ready” after smoke succeeds — not merely after `docker compose up`.

### 7. Stop

```bash
bash scripts/stop-vision.sh
```

---

## Environment contract

| variable | meaning | production value |
|---|---|---|
| `DSV4_VISION_ADAPTER` | container path to adapter `.pt` | `/ckpt/merged-004800-5af0c5.pt` (or `/ckpt/latest.pt`) |
| `DSV4_VISION_TOWER` | container path to tower | `/tower/deepencoder_v2_tower.safetensors` |
| `CKPT_DIR` | host dir mounted at `/ckpt` | directory containing the adapter file |
| `TOWER_DIR` | host dir mounted at `/tower` | directory containing the tower file |
| `PLUGIN_DIR` | host plugin tree | this repo’s `plugin/` |
| `HF_CACHE` | host Hugging Face cache | `~/.cache/huggingface` |
| `MASTER_ADDR` / `MASTER_PORT` | vLLM multi-node rendezvous | head fabric IP / `25100` |
| `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` | RoCE device + IP if | host-specific |

Compose always serves:

```text
vllm serve /cache/huggingface/dsv4-0731-vision
  --tensor-parallel-size 2 --nnodes 2
  --kv-cache-dtype nvfp4_ds_mla
  --moe-backend flashinfer_b12x
  --port 8899
```

---

## Optional: 1× Spark with Antirez 2-bit (text-only)

The **vision** stack above needs the full FP8 0731 backbone (~167 GB) and is
validated only at **TP=2 / 2 nodes**. A single DGX Spark (~121–128 GB UMA)
cannot host that FP8 checkpoint for the same path.

If you want **fast text-only** inference on **one** Spark, use Salvatore
Sanfilippo’s (`antirez`) purpose-built engine and GGUF quant — **not** this
vision plugin.

### What this is

| | |
|---|---|
| Engine | [`antirez/ds4`](https://github.com/antirez/ds4) (C, CUDA backend) |
| Quant | [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf) **q2-imatrix** ~81 GB |
| Vision | **none** — no projector splice; text chat only |
| TP | ds4 `--tensor-parallel` is **Metal-only** (Apple). On CUDA/Spark use **single process** |

### Rough steps (optional; not part of the default compose)

```bash
# on the single Spark
git clone https://github.com/antirez/ds4.git ~/ds4
cd ~/ds4
# follow ds4 README for CUDA / DGX Spark build (see ds4/QA_BEFORE_RELEASES.md §CUDA)

# download the 2-bit Flash quant (~81 GB)
./download_model.sh q2-imatrix
# → gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
#   linked as ./ds4flash.gguf

# interactive / server (exact flags: see ds4 --help on your build)
./ds4 -m ds4flash.gguf --ctx 8192 --nothink -p "Reply with exactly: OK"
# or: ./ds4-server ...   # OpenAI-compatible HTTP if enabled in your build
```

### Optional MTP add-on

```bash
./download_model.sh mtp
# use with q2/q4 Flash quants per ds4 docs
```

### What you should **not** expect

- **No vision** with Antirez q2 unless you build a separate GGUF multimodal path
  (not provided here; the FlyCockpit encoder targets **vLLM + HF FP8 0731**).
- **No free TP=2 speedup** from ds4 on Sparks: tensor parallel is Metal-only;
  multi-node ds4 on CUDA is pipeline-style if available, not this playbook’s NCCL TP.
- **No drop-in of this HF adapter** onto the GGUF — different format and embedding
  contract; the adapter is 0731-embedding-space specific.

### When to use which

| goal | path |
|---|---|
| Vision + screenshots on Sparks | **This playbook (2× FP8 + HF encoder)** |
| Max single-box text tok/s, no images | **1× ds4 + Antirez q2** |
| 2× capacity for text only | Two independent ds4 servers + load balancer, **or** vLLM text TP=2 without the vision plugin |

---

## Guards (learned the hard way)

1. **md5 both nodes** after every download/copy of the adapter.  
2. **Never interleave two full reloads** of the 167 GB backbone.  
3. Preflight reads `config` (not `cfg`). Wrong key → silent `tiles=0` layout.  
4. `--limit-mm-per-prompt` counts images in the *prompt history*; default **8**.  
5. Non-interactive SSH often lacks `hf` on `PATH` — use absolute paths or login shells carefully.  
6. vLLM is **compose-only** in this recipe: it survives SSH disconnect but **not**
   reboot unless you add a systemd unit.

---

## Compatibility

| base weights | this encoder |
|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` | **yes** |
| preview / DSpark / `nvidia/…-NVFP4` (old line) | **no** |

The projector is precision-independent for quantizations of **the same** 0731
weights, but provenance-independent it is not. Hash `embed.weight` rows before
adopting any new quant.

---

## Related

| resource | role |
|---|---|
| [HF vision release](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision) | **Official tower + final adapter** |
| [HF training history](https://huggingface.co/FlyCockpit/dsv4-vision-training-checkpoints) | Intermediate merges / replicas |
| [HF design/code](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision-adapter) | Training code / SPEC |
| [MiaAI 2× DSpark text recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | Packaging lineage for 2-node Spark launches |
| [Anemll base image](https://ghcr.io/anemll/dspark-vllm-gx10:0.1.1) | vLLM + B12X Spark runtime |

---

## License

Repo-local scripts and docs: see [`LICENSE`](LICENSE).  
Model weights and base images retain their upstream licenses.
