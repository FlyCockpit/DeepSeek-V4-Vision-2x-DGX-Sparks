#!/usr/bin/env python3
"""Round 3: is the real-screenshot failure "capability absent off-distribution",
or "capability present but not TRIGGERED by the bare question"?

Same image, same element, three prompt strengths. If the nudge unlocks a
well-formed coordinate, the capability transferred and only the trigger is
distribution-bound -- which a system prompt fixes for free. If nothing unlocks
it, the capability genuinely does not reach real screenshots.
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

NUDGE = ("Answer with only the click point as (x, y), normalised to 0-999. "
         "Do not write any other text.")

CASES = [
    ("014.png", "Where is the News navitem?", None, "bare"),
    ("014.png", "Where is the News navitem? " + NUDGE, None, "inline-nudge"),
    ("014.png", "Where is the News navitem?", NUDGE, "system-prompt"),
    ("000.png", "Where is the Contact navitem?", NUDGE, "system-prompt"),
    ("010.png", "Where is the Volunteer navitem?", NUDGE, "system-prompt"),
]
EXPECT = {("014.png", "News"): (506, 482), ("000.png", "Contact"): (573, 29),
          ("010.png", "Volunteer"): (256, 51)}


def ask(png: Path, question: str, system: str | None, max_tokens=96):
    b64 = base64.b64encode(png.read_bytes()).decode()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": question},
    ]})
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps({"model": MODEL, "max_tokens": max_tokens,
                         "temperature": 0, "messages": msgs}).encode(),
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


for img, q, system, where in CASES:
    ans, err = ask(BUNDLE / img, q, system)
    if err:
        print(f"\n{img} [{where}] REQUEST FAILED {err}")
        continue
    m = COORD.search(ans)
    label = next(k[1] for k in EXPECT if k[0] == img and k[1] in q)
    exp = EXPECT[(img, label)]
    print(f"\n{img} [{where:13s}] {label}   expect~{exp}")
    print(f"   -> {ans[:200]!r}")
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        d = ((x - exp[0]) ** 2 + (y - exp[1]) ** 2) ** 0.5
        print(f"   COORD ({x}, {y})  d={d:.1f}")
    else:
        print("   NO COORDINATE")
