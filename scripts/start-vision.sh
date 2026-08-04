#!/usr/bin/env bash
# Start TP=2 vision stack. Run on the HEAD only.
# Worker first (headless), then head — same order as MiaAI DSpark recipes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.vision}"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT/docker-compose.vision.yml}"
PROJECT="${PROJECT_NAME:-dsv4vision}"
API_URL="${API_URL:-http://127.0.0.1:8899/v1/models}"
CHAT_URL="${CHAT_URL:-http://127.0.0.1:8899/v1/chat/completions}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-80}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE — copy .env.vision.example and edit." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${MASTER_ADDR:?MASTER_ADDR must be set}"
: "${WORKER_HOST:?WORKER_HOST must be set}"
: "${HEAD_FABRIC_IP:?HEAD_FABRIC_IP must be set}"
: "${WORKER_FABRIC_IP:?WORKER_FABRIC_IP must be set}"
: "${HF_CACHE:?}"
: "${PLUGIN_DIR:?}"
: "${TOWER_DIR:?}"
: "${CKPT_DIR:?}"
: "${NCCL_IB_HCA:?}"
: "${NCCL_SOCKET_IFNAME:?}"

WORKER_DIR="${WORKER_DIR:-$ROOT}"
# Worker may use different home usernames; override paths when needed.
WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
WORKER_PLUGIN_DIR="${WORKER_PLUGIN_DIR:-$PLUGIN_DIR}"
WORKER_TOWER_DIR="${WORKER_TOWER_DIR:-$TOWER_DIR}"
WORKER_CKPT_DIR="${WORKER_CKPT_DIR:-$CKPT_DIR}"
ADAPTER_IN_CONTAINER="${DSV4_VISION_ADAPTER:-/ckpt/merged-004800-5af0c5.pt}"
MASTER_PORT="${MASTER_PORT:-25100}"
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

write_env() {
  local rank="$1" fabric_ip="$2" headless="$3" out="$4"
  local hf="$5" plugin="$6" tower="$7" ckpt="$8"
  cat >"$out" <<EOF
HF_CACHE=$hf
PLUGIN_DIR=$plugin
TOWER_DIR=$tower
CKPT_DIR=$ckpt
DSPARK_VLLM_IMAGE=${DSPARK_VLLM_IMAGE:-dsv4-vision-vllm:0.1.1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-vision}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8899}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
MM_IMAGE_LIMIT=${MM_IMAGE_LIMIT:-8}
VLLM_HOST_IP=$fabric_ip
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
NODE_RANK=$rank
HEADLESS=$headless
CHAT_TEMPLATE_FLAG=${CHAT_TEMPLATE_FLAG:-}
DSV4_VISION_ADAPTER=$ADAPTER_IN_CONTAINER
NCCL_IB_HCA=$NCCL_IB_HCA
NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
EOF
}

echo "== sync compose + plugin path assumptions =="
# Ensure worker has the same tree; user is responsible for clone/rsync of the repo.
"${SSH[@]}" "$WORKER_HOST" "test -d '$WORKER_DIR/plugin' && test -f '$WORKER_DIR/docker-compose.vision.yml'" \
  || { echo "Worker missing $WORKER_DIR — clone/rsync this repo to the worker first." >&2; exit 1; }

echo "== refresh model dirs (architecture symlink tree) =="
python3 "$PLUGIN_DIR/make_vision_model_dir.py" | tail -5
"${SSH[@]}" "$WORKER_HOST" "python3 '$WORKER_PLUGIN_DIR/make_vision_model_dir.py' | tail -5"

echo "== write per-node env files =="
write_env 0 "$HEAD_FABRIC_IP" "" "$ROOT/.env.vision.runtime" \
  "$HF_CACHE" "$PLUGIN_DIR" "$TOWER_DIR" "$CKPT_DIR"
write_env 1 "$WORKER_FABRIC_IP" "1" /tmp/.env.vision.worker.$$ \
  "$WORKER_HF_CACHE" "$WORKER_PLUGIN_DIR" "$WORKER_TOWER_DIR" "$WORKER_CKPT_DIR"
scp -q /tmp/.env.vision.worker.$$ "${WORKER_HOST}:${WORKER_DIR}/.env.vision"
cp "$ROOT/.env.vision.runtime" "$ROOT/.env.vision.active"
rm -f /tmp/.env.vision.worker.$$

echo "== start WORKER (rank 1, headless) =="
"${SSH[@]}" "$WORKER_HOST" \
  "cd '$WORKER_DIR' && docker compose -p '$PROJECT' -f docker-compose.vision.yml --env-file .env.vision up -d"

sleep 5

echo "== start HEAD (rank 0) =="
cd "$ROOT"
docker compose -p "$PROJECT" -f "$COMPOSE_FILE" --env-file .env.vision.runtime up -d

echo "== wait for API =="
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null 2>&1; then
    echo "API is up: $API_URL"
    curl -fsS --max-time 5 "$API_URL" || true
    echo
    echo "Running minimal text chat..."
    curl -fsS --max-time 120 "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-vision}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: VISION_OK\"}],\"max_tokens\":16,\"temperature\":0}" \
      | head -c 500
    echo
    echo "START_OK — only now is the stack ready to announce."
    exit 0
  fi
  sleep "$WAIT_SECONDS"
done

echo "Timed out waiting for API. Head logs:" >&2
docker compose -p "$PROJECT" -f "$COMPOSE_FILE" --env-file .env.vision.runtime logs --tail=80 >&2 || true
echo "Worker logs:" >&2
"${SSH[@]}" "$WORKER_HOST" \
  "cd '$WORKER_DIR' && docker compose -p '$PROJECT' -f docker-compose.vision.yml --env-file .env.vision logs --tail=80" >&2 || true
exit 1
