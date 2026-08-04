#!/usr/bin/env python3
"""Control for round 3: does the NUDGE cause the (500, 500) collapse, or do real
screenshots cause it?

Same nudge, same decoding, but on the synthetic renders that answered correctly
without it. If synthetic-plus-nudge stays varied and accurate, the collapse is a
property of real images, not of the instruction -- which is the difference
between "the prompt broke it" and "grounding does not transfer".

Also records finish_reason, because one earlier real-screenshot answer came back
as an empty string and an empty string can mean either a length cutoff inside a
reasoning block or a genuinely empty completion.
"""
import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8899"
MODEL = "deepseek-v4-flash-0731-vision"
COORD = re.compile(r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")
LOCATE = re.compile(r"^(Where is the |Click the |What are the coordinates of the )")
NUDGE = ("Answer with only the click point as (x, y), normalised to 0-999. "
         "Do not write any other text.")


def ask(png: Path, question: str, system=None, max_tokens=96):
    b64 = base64.b64encode(png.read_bytes()).decode()
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": question}]}]
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({"model": MODEL, "max_tokens": max_tokens,
                         "temperature": 0, "messages": msgs}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return None, None, f"HTTP {e.code}: {e.read()[:200].decode()}"
    except Exception as e:                                        # noqa: BLE001
        return None, None, f"{type(e).__name__}: {e}"
    ch = body["choices"][0]
    txt = ch["message"].get("content") or ""
    i = txt.rfind("</think>")
    return (txt[i + len("</think>"):].strip() if i >= 0 else txt.strip()), \
        ch.get("finish_reason"), None


root = Path.home() / "probe2048"
rows = [json.loads(l) for l in (root / "meta.jsonl").open()]
seen = set()
print("=== synthetic 2048, WITH the same nudge ===")
for rec in rows:
    for q, gold in rec["qa"]:
        if not LOCATE.match(q) or rec["image"] in seen:
            continue
        seen.add(rec["image"])
        ans, fin, err = ask(root / rec["image"], q, NUDGE)
        if err:
            print(f"  ! {q[:50]:52s} {err}")
            break
        m, gm = COORD.search(ans), COORD.search(gold)
        if not m:
            print(f"  ? {q[:50]:52s} gold={gold:12s} NO COORD -> {ans[:50]!r}")
            break
        x, y = int(m.group(1)), int(m.group(2))
        gx, gy = int(gm.group(1)), int(gm.group(2))
        print(f"  + {q[:50]:52s} gold=({gx:3d},{gy:3d}) got=({x:3d},{y:3d}) "
              f"d={((x-gx)**2+(y-gy)**2)**0.5:6.1f}  [{fin}]")
        break

print("\n=== the empty answer, re-run with finish_reason ===")
img = Path.home() / "bundle/eval_v1_bundle/images/000.png"
for mt in (96, 384):
    ans, fin, err = ask(img, "Where is the Contact navitem?", None, mt)
    print(f"  max_tokens={mt:4d} finish={fin} len={len(ans or '')} -> {(ans or '')[:90]!r}")
