#!/usr/bin/env bash
# Download backbone + official vision encoder onto this node.
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

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
ASSETS="${ASSETS_DIR:-$HOME/dsv4-vision-assets}"
TOWER_DIR="${TOWER_DIR:-$ASSETS/tower}"
CKPT_DIR="${CKPT_DIR:-$ASSETS/adapter}"
VISION_REPO="${VISION_REPO:-FlyCockpit/DeepSeek-V4-Flash-0731-vision}"
BACKBONE_REPO="${BACKBONE_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}"

ADAPTER_MD5_EXPECT="${ADAPTER_MD5_EXPECT:-d9b3b3bda8f790ecf7cd5a98e6fb93a5}"
TOWER_MD5_EXPECT="${TOWER_MD5_EXPECT:-2d5dba626d816cc367d28b32e744830e}"

export HF_HOME="${HF_HOME:-$HF_CACHE}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# Optional auth for private HF assets (public release does not need a token).
# Prefer: export HF_TOKEN=...  or  HF_TOKEN_FILE=/path/to/token
if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
  export HF_TOKEN
fi

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif [[ -x "$HOME/.local/bin/hf" ]]; then
  HF=("$HOME/.local/bin/hf")
else
  HF=(python3 -m huggingface_hub.cli.hf)
fi

mkdir -p "$TOWER_DIR" "$CKPT_DIR" "$HF_CACHE" "$ASSETS"

echo "== 1/3 backbone: $BACKBONE_REPO into HF cache =="
"${HF[@]}" download "$BACKBONE_REPO"

python3 - "$HF_CACHE" "$BACKBONE_REPO" <<'PY'
import glob, os, sys
cache, repo = sys.argv[1], sys.argv[2]
# hub layout: models--org--name
safe = "models--" + repo.replace("/", "--")
pat = os.path.join(cache, "hub", safe, "snapshots", "*")
snaps = sorted(glob.glob(pat))
if not snaps:
    sys.exit(f"No snapshot under {pat}. Check HF_HOME/HF_CACHE and retry download.")
print("snapshot:", snaps[0])
n = len([n for n in os.listdir(snaps[0]) if n.endswith(".safetensors")])
print("shards:", n)
if n != 48:
    sys.exit(f"expected 48 safetensor shards, got {n}")
print("BACKBONE_OK")
PY

echo "== 2/3 vision encoder: $VISION_REPO =="
"${HF[@]}" download "$VISION_REPO" \
  tower/deepencoder_v2_tower.safetensors \
  --local-dir "$ASSETS"
"${HF[@]}" download "$VISION_REPO" \
  adapter/merged-004800-5af0c5.pt \
  adapter/latest.pt \
  --local-dir "$ASSETS"

ADAPTER_FILE="$ASSETS/adapter/merged-004800-5af0c5.pt"
TOWER_FILE="$ASSETS/tower/deepencoder_v2_tower.safetensors"
[[ -f "$ADAPTER_FILE" ]] || { echo "missing $ADAPTER_FILE" >&2; exit 1; }
[[ -f "$TOWER_FILE" ]] || { echo "missing $TOWER_FILE" >&2; exit 1; }

# If .env points CKPT_DIR/TOWER_DIR elsewhere, mirror files there
if [[ "$(realpath "$CKPT_DIR")" != "$(realpath "$(dirname "$ADAPTER_FILE")")" ]]; then
  mkdir -p "$CKPT_DIR"
  cp -n "$ADAPTER_FILE" "$CKPT_DIR/" || cp "$ADAPTER_FILE" "$CKPT_DIR/"
  cp -n "$ASSETS/adapter/latest.pt" "$CKPT_DIR/" 2>/dev/null || \
    ln -sfn merged-004800-5af0c5.pt "$CKPT_DIR/latest.pt"
  ADAPTER_FILE="$CKPT_DIR/merged-004800-5af0c5.pt"
fi
if [[ "$(realpath "$TOWER_DIR")" != "$(realpath "$(dirname "$TOWER_FILE")")" ]]; then
  mkdir -p "$TOWER_DIR"
  cp -n "$TOWER_FILE" "$TOWER_DIR/" || cp "$TOWER_FILE" "$TOWER_DIR/"
  TOWER_FILE="$TOWER_DIR/deepencoder_v2_tower.safetensors"
fi

echo "== 3/3 md5 verify =="
adapter_md5=$(md5sum "$ADAPTER_FILE" | awk '{print $1}')
tower_md5=$(md5sum "$TOWER_FILE" | awk '{print $1}')
echo "adapter $ADAPTER_FILE"
echo "  md5 $adapter_md5 (expect $ADAPTER_MD5_EXPECT)"
echo "tower   $TOWER_FILE"
echo "  md5 $tower_md5 (expect $TOWER_MD5_EXPECT)"
[[ "$adapter_md5" == "$ADAPTER_MD5_EXPECT" ]] || { echo "ADAPTER MD5 MISMATCH" >&2; exit 1; }
[[ "$tower_md5" == "$TOWER_MD5_EXPECT" ]] || { echo "TOWER MD5 MISMATCH" >&2; exit 1; }
ln -sfn "$(basename "$ADAPTER_FILE")" "$(dirname "$ADAPTER_FILE")/latest.pt"

echo "== build dsv4-0731-vision model dir =="
PLUGIN="${PLUGIN_DIR:-$ROOT/plugin}"
export HOME
# make_vision_model_dir uses ~/.cache/huggingface; align HF_CACHE if custom
if [[ "$(realpath "$HF_CACHE")" != "$(realpath "$HOME/.cache/huggingface")" ]]; then
  export HF_HOME="$HF_CACHE"
  # script hardcodes expanduser("~/.cache/huggingface"); ensure symlink or instruct
  mkdir -p "$HOME/.cache"
  if [[ ! -e "$HOME/.cache/huggingface" ]]; then
    ln -sfn "$HF_CACHE" "$HOME/.cache/huggingface"
  fi
fi
python3 "$PLUGIN/make_vision_model_dir.py"

echo "DOWNLOAD_OK"
echo "  adapter -> $ADAPTER_FILE"
echo "  tower   -> $TOWER_FILE"
echo "  CKPT_DIR should be: $(dirname "$ADAPTER_FILE")"
echo "  TOWER_DIR should be: $(dirname "$TOWER_FILE")"
