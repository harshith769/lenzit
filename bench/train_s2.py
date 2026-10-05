"""Train the final S2 head on ALL of FraudBench and save it (tiny .npz, no pickle).

Training set: every real photo once + every fake twice, as generated (PNG) and compression-matched
(JPEG at real-photo qualities), so the head handles both upload paths. Reported numbers come from
bench/evaluate.py (leave-one-generator-out); this script only fits the deployed model.
Usage:
    python -m bench.train_s2 bench/labels_fraudbench.csv bench/features/fraudbench_clip_l14.npz \
        bench/features/fraudbench_clip_l14_jpegmatched.npz models/s2_head_v1.npz
"""
import csv
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

C = 0.5


def load(path):
    d = np.load(path)
    return dict(zip(d["image_id"].tolist(), d["X"]))


def main(labels, base_npz, matched_npz, out):
    with open(labels, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("role", "evidence") == "evidence"]
    base, matched = load(base_npz), load(matched_npz)
    X, y = [], []
    for r in rows:
        fake = r["label"] == "fake"
        X.append(base[r["image_id"]]); y.append(fake)
        if fake:
            X.append(matched[r["image_id"]]); y.append(True)
    X, y = np.stack(X), np.array(y)
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(scaler.transform(X), y)
    meta = {
        "version": "s2-v1-2026-10-05",
        "backbone": "OpenAI CLIP ViT-L/14 (quickgelu), 5 native 224px crops of the canonical image",
        "trained_on": "FraudBench, all 6 generators; fakes as generated + compression-matched",
        "n_real": int((~y).sum()), "n_fake": int(y.sum()), "C": C,
        "licence": "CC BY-NC-SA 4.0 (inherits FraudBench); retrain without FraudBench for any commercial use",
    }
    np.savez(out, coef=lr.coef_[0].astype(np.float32), intercept=np.float32(lr.intercept_[0]),
             mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             meta=np.array(json.dumps(meta)))
    auc = roc_auc_score(y, lr.predict_proba(scaler.transform(X))[:, 1])
    print(f"saved {out}: {meta['n_real']} real + {meta['n_fake']} fake rows | "
          f"in-sample AUROC {auc:.3f} (sanity check only, not a result)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
