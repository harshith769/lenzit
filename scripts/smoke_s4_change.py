"""S4 change-map smoke test.
Reference = your photo. Evidence = the same photo slightly rotated/scaled/cropped, with a painted
dark 'damage' blob. The change map should peak on the blob. Also checks a different object.
Usage: PYTHONPATH=. python scripts/smoke_s4_change.py data/test.jpeg data/other.jpg
"""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline import s4_reference as s4
from pipeline.canonical import canonicalise

ref = canonicalise(Image.open(sys.argv[1]))
ev = ref.rotate(4, resample=Image.BICUBIC).crop((40, 30, 1000, 750)).resize(ref.size, Image.BICUBIC)
bx, by = 620, 300
d = ImageDraw.Draw(ev)
d.ellipse((bx, by, bx + 90, by + 50), fill=(40, 30, 25))
d.line((bx - 40, by + 25, bx + 140, by + 60), fill=(30, 25, 20), width=6)
ev = canonicalise(ev)

s4.run(ev, ref)  # warm-up (model load)
r = s4.run(ev, ref, "test")
m = r.artifacts["change_mask"]
blob = np.zeros(m.shape, bool)
blob[by - 5:by + 70, bx - 45:bx + 145] = True
print(f"SAME ITEM + DAMAGE  score {r.score} | same_item {r.artifacts['same_item']} | aligned {r.artifacts['aligned']} "
      f"({r.artifacts['inliers']} inliers, {r.artifacts['method']}) | {r.latency_ms} ms")
print(f"  mean change on the damage {m[blob].mean():.3f} vs elsewhere {m[~blob].mean():.3f}")

heat = cv2.applyColorMap((np.clip(m / (m.max() + 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
Image.fromarray((0.55 * np.array(ev, np.float32) + 0.45 * heat).astype(np.uint8)).save("data/s4_debug.png")
print("  saved data/s4_debug.png (open with: explorer.exe data)")

if len(sys.argv) > 2:
    other = canonicalise(Image.open(sys.argv[2]))
    r2 = s4.run(other, ref, "test")
    print(f"DIFFERENT OBJECT   score {r2.score} | same_item {r2.artifacts['same_item']} | "
          f"aligned {r2.artifacts['aligned']} ({r2.artifacts['method']}) | {r2.reason}")
