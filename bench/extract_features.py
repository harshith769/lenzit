"""Canonicalise every image listed in a labels CSV and save frozen CLIP features (S2 input).

Usage (from repo root):
    python -m bench.extract_features bench/labels.csv bench/features/clip_l14.npz
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.canonical import canonicalise
from pipeline.clip_features import DEVICE, clip_features


def main(labels_csv: str, out_path: str) -> None:
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    ids, feats, t0 = [], [], time.time()
    for i, r in enumerate(rows, 1):
        img = canonicalise(Image.open(r["path"]))
        feats.append(clip_features(img))
        ids.append(r["image_id"])
        if i % 25 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} images, {time.time() - t0:.1f}s")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, image_id=np.array(ids), X=np.stack(feats).astype(np.float32))
    print(f"saved {len(ids)} x {feats[0].shape[0]} features to {out_path} (device: {DEVICE})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
