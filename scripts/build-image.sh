#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE="${DSPARK_VLLM_IMAGE:-dsv4-vision-vllm:0.1.1}"

cd "$ROOT"
echo "Building $IMAGE from $ROOT/Dockerfile (context = repo root)"
docker build -t "$IMAGE" -f Dockerfile .
docker image inspect "$IMAGE" --format 'OK {{.Id}} {{.RepoTags}}'
