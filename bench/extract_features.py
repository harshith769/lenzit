"""Canonicalise every image listed in a labels CSV and save frozen CLIP features (S2 input).

Works in batches of BATCH images: 8 threads decode and canonicalise one batch, then the GPU
encodes it in a single call. At most BATCH decoded images are in memory at any time.
Unreadable files are skipped and reported instead of stopping the run.

Options:
  --jpeg-match-fakes  before canonicalisation, JPEG-compress every FAKE at a quality drawn from the
                      real images' own JPEG qualities (estimated from their quantisation tables), so
                      both classes share a lossy compression history (shortcut control, doc section 10)
  --only-label LABEL  extract only rows with this label (e.g. fake) to reuse other features

Usage (from repo root):
    python -m bench.extract_features bench/labels.csv bench/features/clip_l14.npz [options]
"""
import argparse
import csv
import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.canonical import canonicalise
from pipeline.clip_features import DEVICE, clip_features_batch

BATCH = 16
IJG_LUMA_Q50_MEAN = 57.625  # mean of the standard IJG luminance table at quality 50


def quality_from_qtable(table) -> int:
    scale = float(np.mean(table)) / IJG_LUMA_Q50_MEAN * 100
    q = (200 - scale) / 2 if scale < 100 else 5000 / scale
    return int(np.clip(round(q), 10, 100))


def real_jpeg_qualities(rows) -> np.ndarray:
    qs = []
    for r in rows:
        if r["label"] == "fake":
            continue
        try:
            with Image.open(r["path"]) as im:
                q = getattr(im, "quantization", None) or {}
                if 0 in q:
                    qs.append(quality_from_qtable(q[0]))
        except Exception:
            pass
    return np.array(qs)


def matched_quality(image_id: str, qualities: np.ndarray) -> int:
    seed = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16)  # deterministic per image
    return int(np.random.default_rng(seed).choice(qualities))


def _load(row, qualities):
    try:
        img = Image.open(row["path"])
        if qualities is not None and row["label"] == "fake":
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=matched_quality(row["image_id"], qualities))
            buf.seek(0)
            img = Image.open(buf)
        return row, canonicalise(img), None
    except Exception as e:  # corrupt / unsupported file
        return row, None, repr(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--jpeg-match-fakes", action="store_true")
    ap.add_argument("--only-label")
    a = ap.parse_args()
    with open(a.labels, newline="") as f:
        rows = list(csv.DictReader(f))
    qualities = None
    if a.jpeg_match_fakes:
        qualities = real_jpeg_qualities(rows)
        print(f"real JPEG quality: n={len(qualities)} median={np.median(qualities):.0f} "
              f"p10={np.percentile(qualities, 10):.0f} p90={np.percentile(qualities, 90):.0f}")
    if a.only_label:
        rows = [r for r in rows if r["label"] == a.only_label]
    ids, feats, failed, t0 = [], [], [], time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for b, start in enumerate(range(0, len(rows), BATCH)):
            batch = rows[start:start + BATCH]
            results = list(pool.map(lambda r: _load(r, qualities), batch))  # only this batch in memory
            ok = [(r, img) for r, img, err in results if err is None]
            failed += [(r["image_id"], err) for r, img, err in results if err]
            if ok:
                feats.extend(clip_features_batch([img for _, img in ok]))
                ids.extend(r["image_id"] for r, _ in ok)
            done = min(start + BATCH, len(rows))
            if b % 16 == 15 or done == len(rows):
                rate = done / (time.time() - t0)
                print(f"{done}/{len(rows)} images | {rate:.1f} img/s | ~{(len(rows) - done) / rate / 60:.1f} min left", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, image_id=np.array(ids), X=np.stack(feats).astype(np.float32))
    print(f"saved {len(ids)} x {feats[0].shape[0]} features to {a.out} (device: {DEVICE})")
    if failed:
        print(f"{len(failed)} images failed; first 3: {failed[:3]}")


if __name__ == "__main__":
    main()
