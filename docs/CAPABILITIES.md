# Capabilities

Honest results for the production encoder (**step 4800**,
`merged-004800-5af0c5.pt`, md5 `d9b3b3bda8f790ecf7cd5a98e6fb93a5`) served with
`deepseek-ai/DeepSeek-V4-Flash-0731` FP8 on 2× DGX Spark.

**TL;DR:** great at *understanding* screenshots; not a finished click-agent.

## Layout

- `config.tiles = 2`
- Image tokens per image: **257** (≤1536 px), **769**, or **1281** (aspect-aware tiling)
- Image token id: **129279** (`<｜image｜>`) — already reserved in the text-only 0731 checkpoint
- Adapter params: **20,459,520**
- Tower: frozen DeepEncoderV2, 896-d features → projector → 4096-d LM space

## What works

| probe | result |
|---|---|
| Text chat | Coherent; smoke `VISION_OK` returns exact string |
| Image caption / UI description | Works on GUI-like images (validated live) |
| Multi-image follow-up | Supported; keep `--limit-mm-per-prompt` high enough (default 8) — history replays images |
| Synthetic GUI coordinate grounding | **20/20** bare `(x,y)` in 0–999; **16/20** within 100 of ground truth (untiled + tiled sizes) |

## What does not transfer

| probe | result |
|---|---|
| Real webpage screenshots, bare “where is X?” | Usually HTML/prose, not coordinates; rare wrong coordinate |
| Format nudge on real pages (“only (x,y) 0–999”) | Restores **format**, not grounding — often constant **(500, 500)** |
| Same nudge on synthetic GUIs | Still varied/accurate — collapse is real-image specific |

Caveat: the real-image negative used a small n on pages biased toward HTML
generation. **ScreenSpot-v2** (or stronger) has not been run on this merge.

## Production claims

Safe to say:

- “Vision-enabled DeepSeek-V4-Flash-0731 on 2× DGX Spark”
- “Understands screenshots and UI layouts in natural language”
- “In-distribution synthetic GUI pointing works”

Do **not** say without further eval:

- “Production computer-use / click-agent ready on arbitrary real UIs”

## Related HF release

https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision
