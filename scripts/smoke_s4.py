"""Smoke test for S4 same_item_score. Usage: python scripts/smoke_s4.py IMAGE [OTHER_IMAGE]"""
import sys
import time

from PIL import Image, ImageOps

from pipeline.canonical import canonicalise
from pipeline.s4_reference import same_item_score

a = canonicalise(Image.open(sys.argv[1]))
cases = {
    "identical": a,
    "mirrored": ImageOps.mirror(a),
    "cropped+rotated": a.crop((150, 100, 900, 700)).rotate(8, expand=True),
}
if len(sys.argv) > 2:
    cases["different object"] = canonicalise(Image.open(sys.argv[2]))
same_item_score(a, a)  # warm-up (loads the model)
for name, b in cases.items():
    t = time.time()
    s = same_item_score(a, b)
    print(f"{name:18s} {s:.3f}   ({(time.time() - t) * 1000:.0f} ms)")
