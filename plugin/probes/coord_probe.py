#!/usr/bin/env python3
"""Is the coordinate capability PRESENT in the shipped artifact, and is the
format right?

This is a capability/format probe, NOT an accuracy benchmark. The questions are
replayed verbatim from the generator's own annotations, so they are exactly the
phrasing the grounding data was written in; ground truth comes free with them and
is printed only as a sanity signal. A wrong coordinate is an expected pass. No
coordinate at all is the finding.

Screens are probed at BOTH canvases: 1024 never trips the >=1536 tiling
threshold, 2048 does, and coordinates in a tiled layout are a different code path
through the splice than coordinates in an untiled one.
"""
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8899"
MODEL = "deepseek-v4-flash-0731-vision"

# The three locate phrasings render_gui.py emits. Anything else in meta.jsonl is
# an identify/enumerate task and is not a coordinate question.
LOCATE = re.compile(r"^(Where is the |Click the |What are the coordinates of the )")
COORD = re.compile(r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")


def post(path, payload, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"__err": f"HTTP {e.code}: {e.read()[:300].decode()}"}
    except Exception as e:                                    # noqa: BLE001
        return {"__err": f"{type(e).__name__}: {e}"}


def ask(png: Path, question: str, max_tokens: int):
    b64 = base64.b64encode(png.read_bytes()).decode()
    r = post("/v1/chat/completions", {
        "model": MODEL, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": question},
        ]}]})
    if "__err" in r:
        return None, r["__err"]
    return r["choices"][0]["message"].get("content") or "", None


def strip_think(txt):
    """Answer = whatever follows the reasoning block, if there is one."""
    i = txt.rfind("</think>")
    return txt[i + len("</think>"):].strip() if i >= 0 else txt.strip()


def probe(root: Path, label: str, per_screen: int, max_tokens: int):
    rows = [json.loads(l) for l in (root / "meta.jsonl").open()]
    asked = parsed = near = 0
    print(f"\n===== {label}  ({root}) =====")
    for rec in rows:
        qa = [(q, a) for q, a in rec["qa"] if LOCATE.match(q)][:per_screen]
        for q, gold in qa:
            txt, err = ask(root / rec["image"], q, max_tokens)
            asked += 1
            if err:
                print(f"  ! {q[:56]:58s} REQUEST FAILED {err}")
                continue
            ans = strip_think(txt)
            m = COORD.search(ans)
            gm = COORD.search(gold)
            gx, gy = (int(gm.group(1)), int(gm.group(2))) if gm else (None, None)
            if not m:
                print(f"  ? {q[:56]:58s} gold={gold:12s} "
                      f"NO COORDINATE -> {ans[:70]!r}")
                continue
            parsed += 1
            x, y = int(m.group(1)), int(m.group(2))
            d = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5 if gm else float("nan")
            near += d <= 100
            bare = "bare" if m.group(0) == ans else "embedded"
            print(f"  + {q[:56]:58s} gold=({gx:3d},{gy:3d}) got=({x:3d},{y:3d}) "
                  f"d={d:6.1f} {bare:8s}")
    print(f"  -- {label}: coordinate emitted in {parsed}/{asked}"
          f"   within-100 {near}/{asked}")
    return asked, parsed


if __name__ == "__main__":
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 192
    a1, p1 = probe(Path.home() / "probe1024", "1024 (untiled)", per, mt)
    a2, p2 = probe(Path.home() / "probe2048", "2048 (tiled)", per, mt)
    print(f"\nCOORDINATE_FORMAT_PRESENT: {p1 + p2 > 0}"
          f"   emitted {p1 + p2}/{a1 + a2}")
