#!/usr/bin/env bash
# CPU-only preflight of the adapter .pt before (or while) serving.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.vision}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

CKPT_DIR="${CKPT_DIR:-$HOME/dsv4-vision-assets/adapter}"
ADAPTER_HOST="${1:-$CKPT_DIR/merged-004800-5af0c5.pt}"
IMAGE="${DSPARK_VLLM_IMAGE:-dsv4-vision-vllm:0.1.1}"
PLUGIN_DIR="${PLUGIN_DIR:-$ROOT/plugin}"

[[ -f "$ADAPTER_HOST" ]] || { echo "missing adapter: $ADAPTER_HOST" >&2; exit 1; }

echo "preflight: $ADAPTER_HOST"
docker run --rm --network none \
  -v "$PLUGIN_DIR:/opt/dsv4-vision-plugin:ro" \
  -v "$(dirname "$ADAPTER_HOST"):/new:ro" \
  --entrypoint python3 "$IMAGE" \
  /opt/dsv4-vision-plugin/preflight_checkpoint.py "/new/$(basename "$ADAPTER_HOST")"
