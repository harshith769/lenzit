"""Cross-dataset test: the S2 head trained on FraudBench, scored on Lenzit-Bench (our own photos).

Per variant (original / stripped / screenshot): AUROC fake vs all real, recall at 5% false alarms on
real damage, false-alarm rate, per-generator recall, with item-level bootstrap CIs (bench.evaluate).
Headline = 'stripped': real and fake share the same JPEG 75 compression, so file format cannot leak.
Usage: python -m bench.eval_cross bench/labels.csv bench/features/lenzit_clip_l14.npz
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from bench.evaluate import bootstrap, metrics
from pipeline.s2_global import _load_head, score_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("features")
    ap.add_argument("--out", default="bench/results/cross_s2.csv")
    a = ap.parse_args()
    d = np.load(a.features)
    feat = dict(zip(d["image_id"].tolist(), d["X"]))
    with open(a.labels, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("role", "evidence") == "evidence" and r["image_id"] in feat]
    scores = np.array([score_features(feat[r["image_id"]]) for r in rows])
    print(f"S2 head {_load_head()['meta']['version']} (trained on FraudBench) -> {a.labels}: "
          f"{len(rows)} evidence images")
    label = np.array([r["label"] for r in rows])
    gen = np.array([r.get("generator") or "none" for r in rows])
    grp = np.array([r["item_id"] for r in rows])
    var = np.array([r.get("variant") or "original" for r in rows])
    for v in ("original", "stripped", "screenshot"):
        idx = var == v
        if not idx.any():
            continue
        n = {k: int((label[idx] == k).sum()) for k in ("real_undamaged", "real_damaged", "fake")}
        print(f"\n[{v}] {n} | {len(set(grp[idx]))} items")
        if n["fake"] == 0 or n["real_damaged"] == 0:
            print("  needs both fakes and real_damaged photos to score")
            continue
        m = metrics(scores[idx], label[idx], gen[idx])
        ci = bootstrap(scores[idx], label[idx], gen[idx], grp[idx])
        for k, val in m.items():
            extra = f"   95% CI [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" if k in ci else ""
            print(f"  {k:26s} {val:.3f}{extra}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "item_id", "label", "generator", "variant", "s2_score"])
        w.writerows((r["image_id"], r["item_id"], r["label"], r.get("generator") or "none",
                     r.get("variant") or "original", round(float(s), 5)) for r, s in zip(rows, scores))
    print(f"\nscores saved to {a.out}")


if __name__ == "__main__":
    main()
