#!/usr/bin/env python3
"""Pre-flight a vision adapter BEFORE serving it. CPU-only; safe while live.

Run against every candidate checkpoint. It surfaces the things that fail
silently rather than loudly:

  * `config.tiles` -- read via the SAME reader the server uses. Reading the
    wrong key (`cfg`) yields None, falls through to 0, and serves a TILED
    adapter with a 257-token layout. No error, just a layout never trained on.
  * merge provenance + `worst_pairwise_cosine` -- replicas drift apart over
    time; merge_adapters refuses below 0.3, but a merge can be legal and still
    poor. Falling cosine is the early warning, so it is printed every time and
    warned on below WARN_COSINE. Trend so far: +0.8062 (850) -> +0.6381 (1000).
    **Cosine is the DECISION signal: it says whether the soup still lives in one
    basin. Step spread is only bookkeeping.**

Note this check is NOT redundant with the orchestrator's own guard, and both
should be kept -- they measure different things:
  * orchestrator: each replica vs THAT replica's own newest checkpoint; skips the
    round above 300. Catches a replica that has STOPPED PUBLISHING.
  * here: publication steps ACROSS replicas. Catches replicas DRIFTING APART.
A fleet can pass one and fail the other in either direction.
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
WARN_STEP_SPREAD = 300   # replicas publish async; flag only when far apart
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
        # The merge is deliberately ASYNCHRONOUS -- replicas publish hundreds of
        # steps apart and that is normal. What matters is HOW far apart, so parse
        # the real per-replica steps and surface the spread. Handles both the new
        # "r0@001000" form and the older "…/adapter-r0-000850.pt" paths.
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
            print(f"  replica steps    : {dict(sorted(steps.items()))}")
            flag = "OK" if spread <= WARN_STEP_SPREAD else \
                f"WIDE (> {WARN_STEP_SPREAD})"
            print(f"  step spread      : {spread} ({lo}..{hi})  {flag}")
            if spread > WARN_STEP_SPREAD:
                print("    NOTE: wide spread is EXPECTED, not a defect. r1 runs ~13%")
                print("    slower than the other replicas, so the gap grows")
                print("    monotonically through normal operation (~210 at 0h,")
                print("    ~282 at 4h, ~354 at 8h). Step-weighting the merge was")
                print("    evaluated and rejected: it moves r1 0.2500 -> 0.2206, too")
                print("    small to justify changing merge semantics mid-run.")
                print("    ** Spread is bookkeeping. COSINE is the decision signal. **")
                print("    Report it; do not treat it as breakage on its own.")
        elif src:
            print("    NOTE: could not parse per-replica steps from merged_from;")
            print("    if these are MAXSTEP-flattened names the provenance may")
            print("    claim a lockstep merge that did not happen.")
    if "merge_weights" in ck:
        print(f"  merge_weights    : {ck['merge_weights']}")
    cos = ck.get("worst_pairwise_cosine")
    if cos is None:
        print("  worst cosine     : ABSENT (pre-303b447 orchestrator merge)")
    else:
        flag = "OK" if cos >= WARN_COSINE else f"LOW (< {WARN_COSINE})"
        print(f"  worst cosine     : {cos:+.4f}  {flag}")
        if cos < WARN_COSINE:
            print("    WARNING: replicas are diverging; a merge can clear the 0.3")
            print("    refusal floor and still be a poor soup. Confirm before serving.")

    # Reconstructed values are deliberately kept OUT of the real fields, so a
    # log-derived cosine is never mistaken for one the merge computed. Surface
    # the note or that context is lost.
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
