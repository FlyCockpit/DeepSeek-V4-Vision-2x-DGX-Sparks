#!/usr/bin/env python3
"""Pre-flight a vision adapter BEFORE serving it. CPU-only; safe while live.

Run against every candidate checkpoint. It surfaces the things that fail
silently rather than loudly:

  * `config.tiles` -- read via the SAME reader the server uses. Reading the
    wrong key (`cfg`) yields None, falls through to 0, and serves a TILED
    adapter with a 257-token layout. No error, just a layout never trained on.
  * merge provenance + `worst_pairwise_cosine` -- multi-source merges can
    drift; cosine is the decision signal, step spread is bookkeeping only.
  * shape/keys/param-count contract, and whether the file is actually DIFFERENT
    from what is being served (guards against "upgrading" to the same bytes).

Usage:  preflight_checkpoint.py <candidate.pt> [currently_served.pt]
"""
import hashlib
import os
import re
import sys

import torch

WARN_COSINE = 0.5        # refusal floor is 0.3; warn well before it
WARN_STEP_SPREAD = 300   # multi-source merges can lag; flag only when far apart
EXPECT_KEYS = {"proj.0.weight", "proj.0.bias", "proj.2.weight",
               "proj.2.bias", "view_seperator"}
EXPECT_PARAMS = 20_459_520
HIDDEN = 4096


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck.get("adapter", ck)
    cfg = ck.get("config") or ck.get("cfg")

    print(f"== {os.path.basename(path)} ==")
    print(f"  md5              : {md5(path)}")
    print(f"  step             : {ck.get('step')}")
    print(f"  top-level keys   : {sorted(ck.keys())}")
    print(f"  config           : {cfg}")

    ok = True
    keys = set(st.keys())
    if keys != EXPECT_KEYS:
        print(f"  KEYS MISMATCH    : {keys ^ EXPECT_KEYS}")
        ok = False
    else:
        print("  keys             : match contract")

    n = sum(v.numel() for v in st.values())
    good = n == EXPECT_PARAMS
    ok &= good
    print(f"  params           : {n:,} {'OK' if good else 'MISMATCH'}")

    sep = st.get("view_seperator")
    sep_ok = sep is not None and tuple(sep.shape) == (HIDDEN,)
    ok &= sep_ok
    print(f"  view_seperator   : {tuple(sep.shape) if sep is not None else None} "
          f"{'OK' if sep_ok else 'MISMATCH'}")
    print(f"  dtypes           : {sorted({str(v.dtype) for v in st.values()})}")

    # --- provenance / drift ------------------------------------------------
    if "merged_from" in ck:
        src = ck["merged_from"]
        print(f"  merged_from      : {src}")
        # Parse source step ids when present (several naming schemes exist).
        steps = {}
        for entry in (src if isinstance(src, (list, tuple)) else [src]):
            s = str(entry)
            m = re.search(r"(r\d+)@0*(\d+)", s) or re.search(
                r"adapter-(r\d+)-0*(\d+)", s)
            if m:
                steps[m.group(1)] = int(m.group(2))
        if steps:
            lo, hi = min(steps.values()), max(steps.values())
            spread = hi - lo
            print(f"  source steps     : {dict(sorted(steps.items()))}")
            flag = "OK" if spread <= WARN_STEP_SPREAD else \
                f"WIDE (> {WARN_STEP_SPREAD})"
            print(f"  step spread      : {spread} ({lo}..{hi})  {flag}")
            if spread > WARN_STEP_SPREAD:
                print("    NOTE: wide spread is common for async multi-source")
                print("    merges. Cosine is the decision signal; spread alone")
                print("    is not breakage.")
        elif src:
            print("    NOTE: could not parse source steps from merged_from.")
    if "merge_weights" in ck:
        print(f"  merge_weights    : {ck['merge_weights']}")
    cos = ck.get("worst_pairwise_cosine")
    if cos is None:
        print("  worst cosine     : ABSENT")
    else:
        flag = "OK" if cos >= WARN_COSINE else f"LOW (< {WARN_COSINE})"
        print(f"  worst cosine     : {cos:+.4f}  {flag}")
        if cos < WARN_COSINE:
            print("    WARNING: sources look divergent; confirm before serving.")

    note = ck.get("provenance_note")
    if note:
        print(f"  provenance_note  : {note}")

    if cfg is None:
        print("  NOTE: config is None/absent -> the server's reader assumes tiles=0")
        print("        (257 tokens/image). If this checkpoint was trained TILED")
        print("        that is silently wrong. NB the `config` KEY may be present")
        print("        with a null VALUE -- test `is None`, not key membership.")
    return ok, st


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ok, cand = report(sys.argv[1])

    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        print()
        _, cur = report(sys.argv[2])
        print("\n== candidate vs currently served ==")
        identical = True
        for k in sorted(EXPECT_KEYS):
            same = torch.equal(cand[k].float(), cur[k].float())
            d = (cand[k].float() - cur[k].float()).abs().mean().item()
            identical &= same
            print(f"  {k:20s} identical={same}  mean|delta|={d:.3e}")
        if identical:
            print("  WARNING: byte-identical to what is already served -- not an upgrade")
            ok = False

    print(f"\nPREFLIGHT_OK: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
