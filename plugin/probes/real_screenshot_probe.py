#!/usr/bin/env python3
"""Coordinate probe on a REAL screenshot, and a plain caption on a real photo.

The synthetic probe answers "is the capability in the artifact". This one asks
the harder question the release actually cares about: does it transfer off the
generator's own distribution? The screenshot is a real rendered web page from the
eval bundle (2560x2168, so it TILES), and the expected coordinates were read off
the image by hand.

Caveat on the expectations: the generator normalises both axes by a SQUARE
canvas, so on a non-square real screenshot the convention is underdetermined.
`expect` below divides each axis by its own extent, which is the reading that
makes coordinates resolution independent. Treat the numbers as a plausibility
check, not a score.
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
BUNDLE = Path.home() / "bundle/eval_v1_bundle/images"
COORD = re.compile(r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")

# (image, question, hand-read expectation in 0-999 or None for free-form)
CASES = [
    ("000.png", "Where is the Contact navitem?",                  (573, 29)),
    ("000.png", "Where is the Home navitem?",                     (329, 29)),
    ("000.png", "Click the Service 2 card. Where should I click?", (497, 862)),
    ("000.png", "What are the coordinates of the About navitem?", (363, 29)),
    ("000.png", "List the navigation items on this screen.",      None),
    ("003.png", "Describe this image in one sentence.",           None),
]


def ask(png: Path, question: str, max_tokens=192):
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


emitted = coord_qs = 0
for img, q, expect in CASES:
    ans, err = ask(BUNDLE / img, q)
    if err:
        print(f"\n{img}  {q}\n   REQUEST FAILED {err}")
        continue
    print(f"\n{img}  {q}")
    print(f"   -> {ans[:300]!r}")
    if expect is None:
        continue
    coord_qs += 1
    m = COORD.search(ans)
    if not m:
        print(f"   NO COORDINATE (expected near {expect})")
        continue
    emitted += 1
    x, y = int(m.group(1)), int(m.group(2))
    d = ((x - expect[0]) ** 2 + (y - expect[1]) ** 2) ** 0.5
    print(f"   parsed=({x}, {y})  hand-read={expect}  d={d:.1f}"
          f"   {'bare' if m.group(0) == ans else 'embedded'}")

print(f"\nREAL_SCREENSHOT_COORDS_EMITTED: {emitted}/{coord_qs}")
