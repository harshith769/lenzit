"""Shortcut check: can fake vs real be told apart from FILE PROPERTIES alone (resolution, bytes per
pixel, JPEG quantisation, format, aspect)? If yes, S2 may partly learn compression history rather
than AI texture, and JPEG augmentation becomes a priority. Grouped 5-fold CV, gradient boosting.
Usage: python -m bench.shortcut_check bench/labels_fraudbench.csv
"""
import csv
import math
import os
import sys

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

NAMES = ["log_pixels", "bytes_per_pixel", "jpeg_q_mean", "is_png", "aspect"]


def props(path):
    with Image.open(path) as im:  # header only, no full decode
        w, h = im.size
        q = getattr(im, "quantization", None) or {}
        qmean = float(np.mean(q[0])) if 0 in q else -1.0
        fmt = im.format
    return [math.log(w * h), os.path.getsize(path) / (w * h), qmean, float(fmt == "PNG"), w / h]


with open(sys.argv[1], newline="") as f:
    rows = [r for r in csv.DictReader(f) if r.get("role", "evidence") == "evidence"]
X = np.array([props(r["path"]) for r in rows])
y = np.array([r["label"] == "fake" for r in rows])
g = np.array([r["item_id"] for r in rows])
s = cross_val_predict(HistGradientBoostingClassifier(), X, y, groups=g, cv=GroupKFold(5),
                      method="predict_proba")[:, 1]
print(f"file-properties-only AUROC (fake vs real): {roc_auc_score(y, s):.3f}")
for name, col in zip(NAMES, X.T):
    print(f"  {name:16s} real median {np.median(col[~y]):9.3f} | fake median {np.median(col[y]):9.3f}")
