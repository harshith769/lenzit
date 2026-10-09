"""Results table + deployed decision policy from data/fb_bench/scores.csv (written by bench.fb_bench score).

Everything is out-of-fold, grouped by product (the 'fold' column). Two settings:
  * SEEN generators:   S2 = s2_seen (head never saw the product).
  * UNSEEN generator:  for each generator g, S2 = s2_lo_<g> (head never saw g), fusion trained without g's fakes,
                       tested on g's fakes + the fold's reals; numbers averaged over the 6 generators.
Operating point: threshold at 5% false alarms on REAL DAMAGE, per variant. CIs: bootstrap over products.
Also: T0 wrong-item detection by S4, and the deployed decision policy -> models/decision_v1.json.
Usage: python -m bench.fb_eval  [--scores data/fb_bench/scores.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bench.evaluate import bootstrap

FA = 0.05
SIGNALS = ["s2", "s3", "s4"]
SETS = {"S2 only": ["s2"], "S3 only": ["s3"], "S4 only": ["s4"]}
# Fusion was dropped on Oct 10 after a fairness check: S3 (TruFor) scores FraudBench's full-frame fakes BELOW real
# photos (AUROC 0.39), so a fitted weight came out negative; S4 separates fakes from intact photos of the same products
# at chance (AUROC 0.54), and its apparent fusion gain came from real-damage rows having damaged references.
# Deployed policy (models/decision_v1.json): S2 sets the risk; S4 wrong-item rule floors "needs verification";
# S3 is shown as a heatmap only; S1 hard rules on top.
S4_WRONG_T = 0.2


def _metrics(score, label):
    fake, dmg = label == "fake", label == "real_damaged"
    thr = np.quantile(score[dmg], 1 - FA)
    return {"auroc": roc_auc_score(fake, score), "recall@5%FA": float(np.mean(score[fake] > thr)),
            "false_alarm_real_damage": float(np.mean(score[dmg] > thr)), "thr": float(thr)}


def _model():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


def _oof(df, cols, s2col, exclude_gen=None, test_gen=None):
    """OOF fused score. Train folds exclude `exclude_gen` fakes; test rows = reals + fakes of test_gen (or all)."""
    X = df[[s2col if c == "s2" else c for c in cols]].to_numpy()
    y = (df["label"] == "fake").to_numpy()
    out = np.full(len(df), np.nan)
    gen, fold = df["generator"].to_numpy(), df["fold"].to_numpy()
    for k in np.unique(fold):
        tr = (fold != k) & ((gen != exclude_gen) if exclude_gen else True)
        te = (fold == k) & ((gen == "none") | (gen == test_gen) if test_gen else True)
        if len(cols) == 1:          # single signal: the raw score, no fitting
            out[te] = X[te, 0]
        else:
            out[te] = _model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def run(a):
    df = pd.read_csv(a.scores)
    gens = sorted(df.loc[df.label == "fake", "generator"].unique())
    n_nan = int(df["s3"].isna().sum())
    df["s3"] = df["s3"].fillna(df["s3"].median())
    df["s4"] = df["s4_score"]
    same = df[df.pair == "same"].reset_index(drop=True)
    lines = [f"# Lenzit-Bench v1 (FraudBench-derived) results\n",
             f"Rows: {len(df)} ({len(same)} same-item evidence, {int((df.pair == 'wrong').sum())} wrong-item T0). "
             f"Products: {same.item_id.nunique()}. Generators: {', '.join(gens)}. S3 missing (filled with median): {n_nan}.\n",
             "Operating point: threshold at 5% false alarms on real-damage photos. 95% CIs: bootstrap over products.\n"]

    for setting in ["seen", "unseen"]:
        lines.append(f"\n## {'Seen generators (product held out)' if setting == 'seen' else 'Unseen generator (leave-one-generator-out, averaged over 6)'}\n")
        lines.append("| Variant | Signals | AUROC | Recall @5% FA | False alarms on real damage |")
        lines.append("|---|---|---|---|---|")
        for v in ["original", "stripped", "jpeg50", "screenshot"]:
            d = same[same.variant == v].reset_index(drop=True)
            if d.empty:
                continue
            lab, grp, gen = d.label.to_numpy(), d.item_id.to_numpy(), d.generator.to_numpy()
            for name, cols in SETS.items():
                if setting == "seen":
                    s = _oof(d, cols, "s2_seen")
                    m = _metrics(s, lab)
                    ci = bootstrap(s, lab, gen, grp, n=300)
                    rec = f"{m['recall@5%FA']:.1%} [{ci['recall@5%FA'][0]:.0%}–{ci['recall@5%FA'][1]:.0%}]"
                    auc = f"{m['auroc']:.3f}"
                else:
                    ms = []
                    for g in gens:
                        s = _oof(d, cols, f"s2_lo_{g}", exclude_gen=g, test_gen=g)
                        keep = ~np.isnan(s)
                        ms.append(_metrics(s[keep], lab[keep]))
                    m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
                    rec, auc = f"{m['recall@5%FA']:.1%} (min {min(x['recall@5%FA'] for x in ms):.0%})", f"{m['auroc']:.3f}"
                lines.append(f"| {v} | {name} | {auc} | {rec} | {m['false_alarm_real_damage']:.1%} |")

    # T0 wrong item: S4 same-item similarity below S4_WRONG_T -> "needs verification"
    t0 = df[df.pair == "wrong"].s4_same.dropna()
    sr = df[(df.pair == "same") & (df.label != "fake") & (df.variant == "original")].s4_same.dropna()
    if len(t0):
        lines.append("\n## T0: right claim, wrong item (S4 same-item check)\n")
        lines.append("| Threshold on same-item similarity | Wrong-item pairs caught | Genuine same-item pairs flagged |")
        lines.append("|---|---|---|")
        for t in [0.5, 0.3, 0.2, 0.15]:
            mark = " (deployed)" if t == S4_WRONG_T else ""
            lines.append(f"| {t}{mark} | {np.mean(t0 < t):.1%} of {len(t0)} | {np.mean(sr < t):.1%} of {len(sr)} |")

    # deployed decision policy: thresholds from out-of-fold S2 scores on real-damage photos (all variants)
    dmg = same.label.to_numpy() == "real_damaged"
    s2 = same.s2_seen.to_numpy()
    t_manip, t_verify = float(np.quantile(s2[dmg], 1 - FA)), float(np.quantile(s2[dmg], 1 - 0.20))
    model = {"version": "decision-v1-2026-10-10",
             "risk": "global_synthetic (S2) score",
             "t_likely_manipulated": round(t_manip, 4), "t_needs_verification": round(t_verify, 4),
             "s4_wrong_item_same_below": S4_WRONG_T,
             "note": "S2 thresholds = 95th / 80th percentile of out-of-fold S2 scores on real-damage photos "
                     "(all variants); S4 rule floors needs_verification; S3 heatmap only",
             "measured_on": "Lenzit-Bench v1 (FraudBench-derived, CC BY-NC-SA 4.0)"}
    Path("models").mkdir(exist_ok=True)
    Path("models/decision_v1.json").write_text(json.dumps(model, indent=2))
    lines.append(f"\n## Deployed decision policy (models/decision_v1.json)\n\n"
                 f"S2 ≥ {t_manip:.3f} → likely manipulated; S2 ≥ {t_verify:.3f} → needs verification; "
                 f"S4 same-item < {S4_WRONG_T} → at least needs verification; S3 shown as heatmap only.")

    out = Path("bench/results/fb_results.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {out} and models/decision_v1.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="data/fb_bench/scores.csv")
    run(ap.parse_args())
