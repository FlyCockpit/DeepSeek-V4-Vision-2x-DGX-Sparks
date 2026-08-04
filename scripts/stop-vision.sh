#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.vision}"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT/docker-compose.vision.yml}"
PROJECT="${PROJECT_NAME:-dsv4vision}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
WORKER_DIR="${WORKER_DIR:-$ROOT}"
SSH=(ssh -o BatchMode=yes)

echo "== stop worker =="
"${SSH[@]}" "$WORKER_HOST" \
  "cd '$WORKER_DIR' && docker compose -p '$PROJECT' -f docker-compose.vision.yml --env-file .env.vision down --remove-orphans" || true

echo "== stop head =="
cd "$ROOT"
if [[ -f "$ROOT/.env.vision.runtime" ]]; then
  docker compose -p "$PROJECT" -f "$COMPOSE_FILE" --env-file .env.vision.runtime down --remove-orphans || true
else
  docker compose -p "$PROJECT" -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down --remove-orphans || true
fi
echo "STOP_OK"
