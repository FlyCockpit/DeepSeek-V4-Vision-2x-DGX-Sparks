#!/usr/bin/env python3
"""Second real-screenshot round, on two pages whose nav bars sit in DIFFERENT
places, so a degenerate constant answer is distinguishable from a real one.

014.png puts its nav strip at mid-height (y~482 normalised), which no top-corner
default would produce by accident. Expectations are hand-read off the rendered
image; each axis is normalised by its own extent.
"""
import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8899"
MODEL = "deepseek-v4-flash-0731-vision"
BUNDLE = Path.home() / "bundle/eval_v1_bundle/images"
COORD = re.compile(r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")

CASES = [
    ("010.png", "Where is the Mission navitem?",   (199, 51)),
    ("010.png", "Where is the Volunteer navitem?", (256, 51)),
    ("014.png", "Where is the Programs navitem?",  (180, 482)),
    ("014.png", "Where is the News navitem?",      (506, 482)),
    ("014.png", "Where is the Events navitem?",    (825, 482)),
    # Same page, same element, phrased the other two trained ways -- separates
    # "cannot locate" from "this phrasing does not trigger it".
    ("014.png", "Click the News navitem. Where should I click?",   (506, 482)),
    ("014.png", "What are the coordinates of the News navitem?",   (506, 482)),
]


def ask(png: Path, question: str, max_tokens=128):
    b64 = base64.b64encode(png.read_bytes()).decode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({
            "model": MODEL, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": question},
            ]}]}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read()[:200].decode()}"
    except Exception as e:                                        # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    txt = body["choices"][0]["message"].get("content") or ""
    i = txt.rfind("</think>")
    return (txt[i + len("</think>"):].strip() if i >= 0 else txt.strip()), None


emitted = 0
for img, q, expect in CASES:
    ans, err = ask(BUNDLE / img, q)
    if err:
        print(f"{img}  {q[:52]:54s} REQUEST FAILED {err}")
        continue
    m = COORD.search(ans)
    if not m:
        print(f"{img}  {q[:52]:54s} exp={str(expect):11s} NO COORD -> {ans[:60]!r}")
        continue
    emitted += 1
    x, y = int(m.group(1)), int(m.group(2))
    d = ((x - expect[0]) ** 2 + (y - expect[1]) ** 2) ** 0.5
    print(f"{img}  {q[:52]:54s} exp={str(expect):11s} got=({x:3d},{y:3d}) "
          f"d={d:6.1f} {'bare' if m.group(0) == ans else 'embedded'}")

print(f"\nROUND2_COORDS_EMITTED: {emitted}/{len(CASES)}")
