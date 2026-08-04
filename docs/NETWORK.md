# Network notes for 2× DGX Spark vision serving

## Planes

| plane | purpose | example |
|---|---|---|
| RoCE fabric | NCCL TP all-reduces + vLLM rendezvous | `192.168.100.0/24` on `enp1s0f1np1` |
| Management / SSH | deploy scripts, scp | NetBird or LAN; prefer real OpenSSH |
| API | clients hit head `:8899` | host network |

`MASTER_ADDR` and `VLLM_HOST_IP` must be **fabric** addresses, not the
management overlay, unless NCCL is intentionally routed there (not validated).

## Reference pair (validated)

| role | fabric | RoCE device | socket ifname | GID index |
|---|---|---|---|---|
| head | 192.168.100.2 | `rocep1s0f1` | `enp1s0f1np1` | 3 |
| worker | 192.168.100.1 | `rocep1s0f1` | `enp1s0f1np1` | 3 |

Discover on your boxes:

```bash
ibdev2netdev
ip -br a
# pick the UP roce* → en* pair with the fabric IPs
```

## Compose requirements

- `network_mode: host`
- `/dev/infiniband` device mount
- `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, `NCCL_IB_ROCE_VERSION_NUM=2`
- Matching `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_GID_INDEX` on both ranks

## SSH from head → worker

`scripts/start-vision.sh` uses `ssh "$WORKER_HOST"`. Configure passwordless
key auth (e.g. `~/.ssh/config` Host alias). If your management SSH is on a
non-standard port, put that in the Host entry — do not hardcode fabric-only
SSH if OpenSSH is not listening there.

## Ports

| port | role |
|---|---|
| 8899 | vLLM OpenAI API (head) |
| 25100 | vLLM multi-node master port |

## Durability

wsmp / reverse proxies (if you use them) can be systemd user services. The
vLLM compose stack in this recipe is **not** — it will not return after reboot
until you re-run `start-vision.sh` (or add your own unit).
