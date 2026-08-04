# Credits

## Vision encoder (FlyCockpit)

- Final merged adapter + packaging:
  [`FlyCockpit/DeepSeek-V4-Flash-0731-vision`](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision)
- Training history:
  [`FlyCockpit/dsv4-vision-training-checkpoints`](https://huggingface.co/FlyCockpit/dsv4-vision-training-checkpoints)

## Backbone

- [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

## Spark vLLM runtime base

- Anemll image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (B12X / GB10 path)

## Two-node DGX Spark packaging lineage

MiaAI-Lab’s DSpark 2× recipes informed the worker-first compose launch pattern:

- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

## Optional 1× 2-bit text path

- Engine: [`antirez/ds4`](https://github.com/antirez/ds4)
- Quant: [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf)

## Foundations

vLLM, FlashInfer, NVIDIA CUDA/NCCL, DeepSeek V4 Flash architecture and weights.

## License notes

Repo-local scripts/docs: see `LICENSE`. Upstream models, base images, and
kernels retain their own licenses and terms.
