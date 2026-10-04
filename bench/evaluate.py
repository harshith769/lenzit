"""Grouped cross-validation for S2 (later: fusion). Source-of-truth doc, section 10.

Out-of-fold (OOF) scores from GroupKFold by item_id; the held-out generator never trains.
Reports recall at 5% false alarms on real damage, the false-alarm rate, AUROC, per-generator
recall, and 95% bootstrap CIs resampled over ITEMS (photos of one item are not independent).

label: real_undamaged | real_damaged | fake      role: reference | evidence (only evidence scored)
Usage (from repo root):
    python -m bench.evaluate bench/features/clip_l14.npz bench/labels.csv --heldout G4
    python -m bench.evaluate --synthetic          # self-test on made-up data
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FA_TARGET = 0.05


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))


def oof_scores(X, y, groups, gen, heldout, n_splits=5):
    scores = np.full(len(y), np.nan)
    k = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        tr = tr[gen[tr] != heldout]  # the held-out generator never trains
        scores[te] = make_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return scores


def metrics(scores, label, gen):
    fake, real_dmg = label == "fake", label == "real_damaged"
    thr = np.quantile(scores[real_dmg], 1 - FA_TARGET)
    out = {
        "auroc": roc_auc_score(fake, scores),
        "recall@5%FA": float(np.mean(scores[fake] > thr)),
        "false_alarm_real_damage": float(np.mean(scores[real_dmg] > thr)),
        "threshold": float(thr),
    }
    for g in sorted(set(gen[fake])):
        out[f"recall[{g}]"] = float(np.mean(scores[fake & (gen == g)] > thr))
    return out


def bootstrap(scores, label, gen, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    items = np.unique(groups)
    rows_of = {g: np.where(groups == g)[0] for g in items}
    keys = ["auroc", "recall@5%FA", "false_alarm_real_damage"]
    samples = {k: [] for k in keys}
    for _ in range(n):
        pick = np.concatenate([rows_of[g] for g in rng.choice(items, len(items), replace=True)])
        lab = label[pick]
        if not ((lab == "fake").any() and (lab == "real_damaged").any()):
            continue
        m = metrics(scores[pick], lab, gen[pick])
        for k in keys:
            samples[k].append(m[k])
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in samples.items()}


def load(features_path, labels_path):
    d = np.load(features_path)
    feat = dict(zip(d["image_id"].tolist(), d["X"]))
    with open(labels_path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("role", "evidence") == "evidence" and r["image_id"] in feat]
    X = np.stack([feat[r["image_id"]] for r in rows])
    label = np.array([r["label"] for r in rows])
    groups = np.array([r["item_id"] for r in rows])
    gen = np.array([r.get("generator") or "none" for r in rows])
    return [r["image_id"] for r in rows], X, label, groups, gen


def synthetic(seed=0):
    """Made-up data shaped like Lenzit-Bench, ONLY to test this script. Its numbers mean nothing."""
    rng = np.random.default_rng(seed)
    ids, X, label, groups, gen = [], [], [], [], []
    def add(i, x, l, item, g):
        ids.append(i); X.append(x); label.append(l); groups.append(item); gen.append(g)
    for item in range(40):
        base = rng.normal(0, 1, 768)
        for j in range(3):
            add(f"i{item}_u{j}", base + rng.normal(0, .5, 768), "real_undamaged", f"item{item}", "none")
        if item < 15:
            for j in range(4):
                add(f"i{item}_d{j}", base + rng.normal(0, .5, 768) + 0.03, "real_damaged", f"item{item}", "none")
        for j, g in enumerate(["G1", "G2", "G3", "G4"] * 2):
            add(f"i{item}_f{j}", base + rng.normal(0, .5, 768) + (0.08 if g != "G4" else 0.04), "fake", f"item{item}", g)
    return ids, np.array(X), np.array(label), np.array(groups), np.array(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", nargs="?")
    ap.add_argument("labels", nargs="?")
    ap.add_argument("--heldout", default="G4")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ids, X, label, groups, gen = synthetic() if a.synthetic else load(a.features, a.labels)
    out = a.out or ("/tmp/lenzit_synthetic_oof.csv" if a.synthetic else "bench/results/oof_s2.csv")

    scores = oof_scores(X, (label == "fake").astype(int), groups, gen, a.heldout)
    m, ci = metrics(scores, label, gen), bootstrap(scores, label, gen, groups)

    print(f"{len(ids)} evidence images | {len(np.unique(groups))} items | held-out generator: {a.heldout}")
    for k, v in m.items():
        extra = f"   95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else ""
        print(f"  {k:26s} {v:.3f}{extra}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "item_id", "label", "generator", "oof_score"])
        w.writerows(zip(ids, groups, label, gen, np.round(scores, 5)))
    print(f"out-of-fold scores saved to {out}")


if __name__ == "__main__":
    main()
