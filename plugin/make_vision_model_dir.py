#!/usr/bin/env python3
"""Create a model directory that is the 0731 snapshot with ONE change:
architectures -> DeepseekV4VisionForCausalLM, so vLLM builds our wrapper.

The directory is created INSIDE the HF cache and its symlinks point at the
*container* path (/cache/huggingface/...), not the host path. That matters for
two reasons:
  - the compose mounts HF_CACHE at /cache/huggingface, so the links resolve
    inside the container;
  - the two nodes have different usernames (christopherkapic vs christopher),
    so host-absolute links would not be portable between them.

Everything is symlinked, so this costs no disk and cannot corrupt the original.
"""
import glob
import json
import os
import pathlib
import sys

HOST_CACHE = os.path.expanduser("~/.cache/huggingface")
CONTAINER_CACHE = "/cache/huggingface"
REPO_DIR = "hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots"

snaps = sorted(glob.glob(os.path.join(HOST_CACHE, REPO_DIR, "*")))
if not snaps:
    sys.exit(f"no snapshot under {os.path.join(HOST_CACHE, REPO_DIR)}")
snap = snaps[0]
sha = os.path.basename(snap)

out = pathlib.Path(HOST_CACHE) / "dsv4-0731-vision"
out.mkdir(parents=True, exist_ok=True)

linked = 0
for name in os.listdir(snap):
    if name == "config.json":
        continue
    dst = out / name
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    # Point at the CONTAINER path; resolves once HF_CACHE is mounted.
    os.symlink(f"{CONTAINER_CACHE}/{REPO_DIR}/{sha}/{name}", dst)
    linked += 1

with open(os.path.join(snap, "config.json")) as fh:
    cfg = json.load(fh)

orig = cfg.get("architectures")
cfg["architectures"] = ["DeepseekV4VisionForCausalLM"]
with open(out / "config.json", "w") as fh:
    json.dump(cfg, fh, indent=2)

print(f"snapshot  : {snap}")
print(f"outdir    : {out}")
print(f"container : {CONTAINER_CACHE}/dsv4-0731-vision")
print(f"symlinked : {linked} entries -> {CONTAINER_CACHE}/{REPO_DIR}/{sha}/")
print(f"arch      : {orig} -> {cfg['architectures']}")
n_shards = len([n for n in os.listdir(out) if n.endswith(".safetensors")])
print(f"shards    : {n_shards}")
assert n_shards == 48, f"expected 48 shards, saw {n_shards}"
print("OK")
