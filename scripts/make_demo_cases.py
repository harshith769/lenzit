"""Build web/demo/ (preloaded one-click cases on the live page) and print what each case returns locally.

Needs in web/demo/ first (our own photos, copied by hand):  ref.jpg  real_dented.jpg  chatgpt_fake.png (untouched download)
Adds automatically:
  chatgpt_fake_screenshot.png  same fake re-saved without metadata (simulates a screenshot: the known limitation)
  fb_fake.jpg + fb_ref.jpg     a FraudBench fake (metadata stripped, JPEG 75) that S2 catches, with its same-product ref
  fb_wrong.jpg + fb_wrong_ref.jpg  a FraudBench wrong-item pair (T0) that S4 flags
Usage (repo root): python scripts/make_demo_cases.py
"""
import io
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.fb_bench import make_variant  # noqa: E402
from pipeline.run import analyse  # noqa: E402

D = Path("web/demo")
need = ["ref.jpg", "real_dented.jpg", "chatgpt_fake.png"]
missing = [n for n in need if not (D / n).exists()]
if missing:
    sys.exit(f"put these in web/demo/ first: {missing}")

# 1. screenshot-like copy of the ChatGPT fake: pixels only, no metadata / Content Credentials
Image.open(D / "chatgpt_fake.png").convert("RGB").save(D / "chatgpt_fake_screenshot.png")

df = pd.read_csv("data/fb_bench/scores.csv")


def first_ok(rows, want, ev_name, ref_name, variant):
    for r in rows.itertuples():
        ev = make_variant(r.ev_path, variant)
        ref = Path(r.ref_path).read_bytes()
        out = analyse(ev, ref, r.claim_text if isinstance(r.claim_text, str) and r.claim_text else "Item arrived damaged")
        if out["verdict"] == want:
            (D / ev_name).write_bytes(ev)
            Image.open(io.BytesIO(ref)).convert("RGB").save(D / ref_name, quality=92)
            return r
    return None


# 2. FraudBench fake that S2 catches (stripped variant), highest S2 first
cand = df[(df.label == "fake") & (df.variant == "stripped") & (df.pair == "same")].sort_values("s2_seen", ascending=False)
fb = first_ok(cand.head(15), "likely_manipulated", "fb_fake.jpg", "fb_ref.jpg", "stripped")
# 3. FraudBench wrong-item pair that S4 flags
wr = df[(df.pair == "wrong")].sort_values("s4_same")
wrong = first_ok(wr.head(10), "needs_verification", "fb_wrong.jpg", "fb_wrong_ref.jpg", "original")

cases = [
    {"title": "Real damage, honest buyer", "note": "Our own photo of a really dented bottle",
     "evidence": "real_dented.jpg", "reference": "ref.jpg", "claim": "Bottle arrived dented on the side"},
    {"title": "ChatGPT fake (credentials intact)", "note": "Dent added by ChatGPT; untouched download",
     "evidence": "chatgpt_fake.png", "reference": "ref.jpg", "claim": "Bottle arrived dented on the side"},
]
if fb is not None:
    cases.append({"title": "AI fake, metadata stripped", "note": f"FraudBench ({fb.generator}), caught by the pixel detector",
                  "evidence": "fb_fake.jpg", "reference": "fb_ref.jpg",
                  "claim": fb.claim_text if isinstance(fb.claim_text, str) and fb.claim_text else "Item arrived damaged"})
if wrong is not None:
    cases.append({"title": "Wrong item in the photo", "note": "Real photo of a different product than the seller's",
                  "evidence": "fb_wrong.jpg", "reference": "fb_wrong_ref.jpg", "claim": "Item arrived damaged"})
cases.append({"title": "Known limitation", "note": "Same ChatGPT fake as a screenshot: credentials gone",
              "evidence": "chatgpt_fake_screenshot.png", "reference": "ref.jpg", "claim": "Bottle arrived dented on the side"})
(D / "cases.json").write_text(json.dumps(cases, indent=2))

print("case -> local verdict (S2, S4 same-item, rules)")
for c in cases:
    ev = (D / c["evidence"]).read_bytes()
    ref = (D / c["reference"]).read_bytes() if c["reference"] else None
    o = analyse(ev, ref, c["claim"])
    s = o["signals"]
    print(f"- {c['title']:36s} {o['verdict']:20s} S2={s['global_synthetic']['score']:.2f} "
          f"S4={s['reference']['score']:.2f} rules={o['rules_applied']}")
print(f"wrote {D/'cases.json'} with {len(cases)} cases; FraudBench fake found: {fb is not None}; wrong-item found: {wrong is not None}")
