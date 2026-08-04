#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.vision}"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT/docker-compose.vision.yml}"
PROJECT="${PROJECT_NAME:-dsv4vision}"
PORT="${VLLM_PORT:-8899}"
API_URL="${API_URL:-http://127.0.0.1:${PORT}/v1/models}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

WORKER_HOST="${WORKER_HOST:-}"
WORKER_DIR="${WORKER_DIR:-$ROOT}"
IMAGE="${DSPARK_VLLM_IMAGE:-dsv4-vision-vllm:0.1.1}"

echo "== head compose =="
cd "$ROOT"
docker compose -p "$PROJECT" -f "$COMPOSE_FILE" ps 2>/dev/null || true
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E 'dsv4|vision|vllm' || true

if [[ -n "$WORKER_HOST" ]]; then
  echo
  echo "== worker compose ($WORKER_HOST) =="
  ssh -o BatchMode=yes "$WORKER_HOST" \
    "cd '$WORKER_DIR' && docker compose -p '$PROJECT' -f docker-compose.vision.yml ps; docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E 'dsv4|vision|vllm' || true" || true
fi

echo
echo "== image =="
docker image inspect "$IMAGE" --format "head $IMAGE {{.Id}}" 2>/dev/null || echo "missing $IMAGE"
if [[ -n "$WORKER_HOST" ]]; then
  ssh -o BatchMode=yes "$WORKER_HOST" \
    "docker image inspect '$IMAGE' --format 'worker $IMAGE {{.Id}}'" 2>/dev/null || true
fi

echo
echo "== adapter env (head container) =="
docker ps -q -f name=dsv4vision | head -1 | xargs -r -I{} docker exec {} printenv DSV4_VISION_ADAPTER DSV4_VISION_TOWER 2>/dev/null || true

echo
echo "== API =="
curl -fsS --max-time 5 "$API_URL" && echo || echo "API not reachable at $API_URL"
