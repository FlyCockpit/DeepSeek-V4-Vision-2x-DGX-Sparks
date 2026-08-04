#!/usr/bin/env bash
# Text smoke by default; pass a PNG/JPEG path for a multimodal smoke.
set -euo pipefail

PORT="${VLLM_PORT:-8899}"
BASE="${API_BASE:-http://127.0.0.1:${PORT}/v1}"
MODEL="${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-vision}"
IMG="${1:-}"

echo "== /v1/models =="
curl -fsS --max-time 10 "$BASE/models"
echo

if [[ -z "$IMG" ]]; then
  echo "== text chat =="
  curl -fsS --max-time 120 "$BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: VISION_OK\"}],\"max_tokens\":16,\"temperature\":0}"
  echo
  exit 0
fi

[[ -f "$IMG" ]] || { echo "not a file: $IMG" >&2; exit 1; }

python3 - "$BASE" "$MODEL" "$IMG" <<'PY'
import base64, json, mimetypes, sys, urllib.request
base, model, path = sys.argv[1:4]
mime = mimetypes.guess_type(path)[0] or "image/png"
b64 = base64.b64encode(open(path, "rb").read()).decode()
body = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image in one short sentence."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }],
    "max_tokens": 64,
    "temperature": 0,
}
req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as r:
    data = json.load(r)
print(json.dumps({
    "image": path,
    "content": data["choices"][0]["message"]["content"],
    "usage": data.get("usage"),
}, indent=2))
PY
