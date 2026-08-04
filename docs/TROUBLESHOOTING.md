# Troubleshooting

## Silent wrong layout (`tiles=0` while serving a tiled adapter)

The checkpoint key is **`config`**, not `cfg`. Reading the wrong key returns
`None`, falls through to tiles=0, and serves a 257-token layout with no error.
Always run `scripts/preflight-adapter.sh` and confirm logs show:

```text
[dsv4-vision] checkpoint config.tiles=2 (/ckpt/merged-004800-5af0c5.pt)
```

## Adapter “upgraded” but nothing changed

md5 both nodes. Intermediate published checksums have been rewritten during
provenance cleanup. Compare to:

- adapter: `d9b3b3bda8f790ecf7cd5a98e6fb93a5`
- tower: `2d5dba626d816cc367d28b32e744830e`

## Second image in a multi-turn chat fails

`--limit-mm-per-prompt` counts images in the **replayed history**, not only the
new turn. Default in this recipe is **8**.

## `hf: command not found` over non-interactive SSH

Non-login SSH often skips `~/.local/bin`. Use absolute paths or:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

`scripts/download-assets.sh` tries common locations.

## NetBird port 22 vs real OpenSSH

On some Spark fleets, port 22 is a NetBird wrapper (`su -` around commands) with
no SFTP — `scp` fails and `nohup` is unreliable. Prefer a real OpenSSH listener
(e.g. port 2222) for automation.

## NCCL hang at bootstrap

- Confirm fabric pings both ways on the RoCE IPs
- `ibdev2netdev` shows the HCA **Up**
- `MASTER_ADDR` is the head **fabric** IP
- Matching GID index (reference used `3`)

## OOM / cannot start on 1 node

Expected: FP8 0731 is ~167 GB; single Spark UMA is ~121–128 GB. Use 2× TP=2, or
the optional **text-only** Antirez q2 path in the README.

## Wrong backbone lineage

If you load preview/DSpark/NVIDIA-NVFP4 weights, the adapter will run but emit
garbage relative to training. Only `deepseek-ai/DeepSeek-V4-Flash-0731` is
supported. Hash embedding rows before trying any third-party quant.

## Container up but API dead

Cold load takes several minutes. Check:

```bash
docker compose -p dsv4vision logs -f
# wait for: Application startup complete.
```

Do not announce readiness until `scripts/smoke-vision.sh` succeeds.
