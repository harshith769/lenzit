"""Canonicalise every image listed in a labels CSV and save frozen CLIP features (S2 input).

Works in batches of BATCH images: 8 threads decode and canonicalise one batch, then the GPU
encodes it in a single call. At most BATCH decoded images are in memory at any time.
Unreadable files are skipped and reported instead of stopping the run.
Usage (from repo root):
    python -m bench.extract_features bench/labels.csv bench/features/clip_l14.npz
"""
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.canonical import canonicalise
from pipeline.clip_features import DEVICE, clip_features_batch

BATCH = 16


def _load(row):
    try:
        return row, canonicalise(Image.open(row["path"])), None
    except Exception as e:  # corrupt / unsupported file
        return row, None, repr(e)


def main(labels_csv: str, out_path: str) -> None:
    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    ids, feats, failed, t0 = [], [], [], time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for b, start in enumerate(range(0, len(rows), BATCH)):
            results = list(pool.map(_load, rows[start:start + BATCH]))  # only this batch in memory
            ok = [(r, img) for r, img, err in results if err is None]
            failed += [(r["image_id"], err) for r, img, err in results if err]
            if ok:
                feats.extend(clip_features_batch([img for _, img in ok]))
                ids.extend(r["image_id"] for r, _ in ok)
            done = min(start + BATCH, len(rows))
            if b % 16 == 15 or done == len(rows):
                rate = done / (time.time() - t0)
                print(f"{done}/{len(rows)} images | {rate:.1f} img/s | ~{(len(rows) - done) / rate / 60:.1f} min left", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, image_id=np.array(ids), X=np.stack(feats).astype(np.float32))
    print(f"saved {len(ids)} x {feats[0].shape[0]} features to {out_path} (device: {DEVICE})")
    if failed:
        print(f"{len(failed)} images failed; first 3: {failed[:3]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
