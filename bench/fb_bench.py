"""Lenzit-Bench v1 (FraudBench-derived): build the test pairs and score S2, S3 and S4 on them, resumably.

Why: our own photo shoot was dropped (Oct 9), so the end-to-end test set is built from FraudBench
(CC BY-NC-SA 4.0) plus scripted attacks. Honesty rules:
  * S2 is scored OUT-OF-FOLD: 5 grouped folds over FraudBench products; a product's images are scored only
    by heads that never saw that product. Per generator g we also train heads WITHOUT g (s2_lo_<g>), so every
    fake can be scored by a model that never saw its generator (leave-one-generator-out).
  * Reference photo = ANOTHER photo from the same review (same product, different shot), for reals,
    real-damage and fakes alike. T0 rows use a different product of the same category (wrong item).
  * Variants (original / stripped JPEG75 / jpeg50 / screenshot) are applied to real and fake evidence alike.

Usage (repo root, GPU on):
  python -m bench.fb_bench select          # writes data/fb_bench/pairs.csv (seconds)
  python -m bench.fb_bench heads           # trains 35 small S2 heads -> data/fb_bench/s2_heads.npz (~15-30 min)
  python -m bench.fb_bench score           # S2+S3+S4 per row -> data/fb_bench/scores.csv (resumable; ~2-3 h)
Options: --products 100 --negatives 100 --wrong 60 --chunk 60 --trufor ~/vendor/TruFor/test_docker/src
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("data/fb_bench")
LABELS = Path("bench/labels_fraudbench.csv")
FEAT_BASE = Path("bench/features/fraudbench_clip_l14.npz")
FEAT_MATCHED = Path("bench/features/fraudbench_clip_l14_jpegmatched.npz")
VARIANTS = ["original", "stripped", "jpeg50", "screenshot"]
N_FOLDS = 5
PAIR_COLS = ["row_id", "src_image_id", "item_id", "category", "label", "generator", "variant", "pair",
             "fold", "claim_text", "ev_path", "ref_path"]


def _safe(s: str) -> str:
    return s.replace(".", "-").replace(" ", "-")


def _read_labels():
    with open(LABELS, newline="") as f:
        return list(csv.DictReader(f))


def _folds(rows):
    """Deterministic product -> fold map (same for heads and bench rows)."""
    items = sorted({r["item_id"] for r in rows})
    rng = random.Random(0)
    rng.shuffle(items)
    return {it: i % N_FOLDS for i, it in enumerate(items)}


# ---------------------------------------------------------------- select
def select(a):
    rows = _read_labels()
    fold = _folds(rows)
    rng = random.Random(0)
    pos, neg = defaultdict(list), defaultdict(list)
    fakes = defaultdict(dict)          # (item_id, stem) -> {gen: row}
    for r in rows:
        stem = Path(r["path"]).stem
        if r["label"] == "real_undamaged":
            pos[r["item_id"]].append(r)
        elif r["label"] == "real_damaged":
            neg[r["item_id"]].append(r)
        else:
            fakes[(r["item_id"], stem)][r["generator"]] = r

    out = []

    def add(ev, ref, label, gen, variant, pair, claim):
        rid = _safe(f"{ev['image_id']}__{variant}" + ("__wrong" if pair == "wrong" else ""))
        out.append({"row_id": rid, "src_image_id": ev["image_id"], "item_id": ev["item_id"],
                    "category": ev["category"], "label": label, "generator": gen, "variant": variant,
                    "pair": pair, "fold": fold[ev["item_id"]], "claim_text": claim or "",
                    "ev_path": ev["path"], "ref_path": ref["path"] if ref else ""})

    # products with >=2 intact photos and at least one faked source photo
    cand = []
    for it, imgs in pos.items():
        if len(imgs) < 2:
            continue
        best = max(imgs, key=lambda r: len(fakes.get((it, Path(r["path"]).stem), {})))
        if fakes.get((it, Path(best["path"]).stem)):
            cand.append((it, best))
    rng.shuffle(cand)
    cand = cand[: a.products]
    chosen_pos = []
    for it, src in cand:
        ref = next(r for r in pos[it] if r is not src)
        chosen_pos.append((src, ref))
        gens = fakes[(it, Path(src["path"]).stem)]
        for v in VARIANTS:
            add(src, ref, "real_undamaged", "none", v, "same", "")
            for g, fr in sorted(gens.items()):
                add(fr, ref, "fake", g, v, "same", fr.get("claim_text", ""))

    negs = [(it, imgs) for it, imgs in neg.items() if len(imgs) >= 2]
    rng.shuffle(negs)
    for it, imgs in negs[: a.negatives]:
        ev, ref = imgs[0], imgs[1]
        for v in VARIANTS:
            add(ev, ref, "real_damaged", "none", v, "same", "")

    # T0: right claim, wrong item (reference from a different product of the same category)
    by_cat = defaultdict(list)
    for src, ref in chosen_pos:
        by_cat[src["category"]].append((src, ref))
    t0 = []
    for cat, lst in by_cat.items():
        for i, (src, _) in enumerate(lst):
            others = [r for j, (_, r) in enumerate(lst) if j != i]
            if others:
                t0.append((src, rng.choice(others)))
    rng.shuffle(t0)
    for src, wrong_ref in t0[: a.wrong]:
        add(src, wrong_ref, "real_undamaged", "none", "original", "wrong", "")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIR_COLS)
        w.writeheader()
        w.writerows(out)
    n = defaultdict(int)
    for r in out:
        n[(r["label"], r["pair"])] += 1
    print(f"products {len(cand)} | negatives {min(len(negs), a.negatives)} (of {len(negs)} with >=2 photos) | "
          f"T0 {min(len(t0), a.wrong)}")
    for k, v in sorted(n.items()):
        print(f"  {k[0]:15s} {k[1]:6s} {v}")
    print(f"total rows {len(out)} -> {OUT / 'pairs.csv'}")


# ---------------------------------------------------------------- heads
def _fit(X, y, C=0.5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return np.concatenate([lr.coef_[0], [lr.intercept_[0]]]).astype(np.float32), sc.mean_, sc.scale_


def heads(a):
    rows = [r for r in _read_labels() if r.get("role", "evidence") == "evidence"]
    fold = _folds(rows)
    base, matched = np.load(FEAT_BASE), np.load(FEAT_MATCHED)
    fb = dict(zip(base["image_id"].tolist(), base["X"]))
    fm = dict(zip(matched["image_id"].tolist(), matched["X"]))
    gens = sorted({r["generator"] for r in rows if r["label"] == "fake"})
    X, y, f_of, g_of = [], [], [], []
    for r in rows:
        if r["image_id"] not in fb:
            continue
        fake = r["label"] == "fake"
        X.append(fb[r["image_id"]]); y.append(fake); f_of.append(fold[r["item_id"]]); g_of.append(r["generator"])
        if fake and r["image_id"] in fm:
            X.append(fm[r["image_id"]]); y.append(True); f_of.append(fold[r["item_id"]]); g_of.append(r["generator"])
    X, y, f_of, g_of = np.stack(X), np.array(y), np.array(f_of), np.array(g_of)
    out = {}
    t0 = time.time()
    for k in range(N_FOLDS):
        for g in ["all"] + gens:
            keep = (f_of != k) & ((g_of != g) if g != "all" else True)
            w, mu, sd = _fit(X[keep], y[keep])
            out[f"f{k}_{g}_w"], out[f"f{k}_{g}_mu"], out[f"f{k}_{g}_sd"] = w, mu.astype(np.float32), sd.astype(np.float32)
            print(f"  head fold {k} without {g:22s} n={keep.sum():6d}  ({time.time() - t0:.0f}s)", flush=True)
    out["gens"] = np.array(gens)
    np.savez(OUT / "s2_heads.npz", **out)
    print(f"saved {len(gens) + 1} x {N_FOLDS} heads -> {OUT / 's2_heads.npz'}")


# ---------------------------------------------------------------- score
def make_variant(path: str, variant: str) -> bytes:
    from PIL import Image, ImageOps
    raw = Path(path).read_bytes()
    if variant == "original":
        return raw
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    buf = io.BytesIO()
    if variant == "stripped":
        img.save(buf, "JPEG", quality=75)              # no exif passed -> metadata removed
    elif variant == "jpeg50":
        img.save(buf, "JPEG", quality=50)
    elif variant == "screenshot":                      # phone screenshot: rescale to 1080 wide + UI bars, PNG
        w, h = img.size
        img = img.resize((1080, max(1, round(h * 1080 / w))), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (1080, img.height + 200), (24, 24, 24))
        canvas.paste(img, (0, 80))
        canvas.save(buf, "PNG")
    return buf.getvalue()


def _s2_scores(x, fold, H):
    out = {}
    for key in ["all"] + list(H["gens"]):
        w, mu, sd = H[f"f{fold}_{key}_w"], H[f"f{fold}_{key}_mu"], H[f"f{fold}_{key}_sd"]
        z = float(((x - mu) / sd) @ w[:-1] + w[-1])
        out["s2_seen" if key == "all" else f"s2_lo_{key}"] = round(1 / (1 + np.exp(-z)), 5)
    return out


def _trufor(images: dict, trufor_src: Path) -> dict:
    """images: row_id -> PIL canonical image. Runs TruFor once on a temp folder; returns row_id -> score."""
    tmp = Path(tempfile.mkdtemp(prefix="lz_tf_"))
    (tmp / "in").mkdir(); (tmp / "out").mkdir()
    for rid, im in images.items():
        im.save(tmp / "in" / f"{rid}.png")
    cmd = [sys.executable, "trufor_test.py", "-gpu", "0", "-in", str(tmp / "in") + "/", "-out", str(tmp / "out") + "/"]
    subprocess.run(cmd, cwd=trufor_src, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = {}
    for f in (tmp / "out").rglob("*.npz"):
        rid = f.name.split(".")[0]
        d = np.load(f)
        res[rid] = round(float(d["score"]), 5) if "score" in d.files else float("nan")
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def score(a):
    from PIL import Image
    from pipeline.canonical import canonicalise
    from pipeline.clip_features import clip_features_batch
    from pipeline import s4_reference

    with open(OUT / "pairs.csv", newline="") as f:
        pairs = list(csv.DictReader(f))
    H = dict(np.load(OUT / "s2_heads.npz"))
    out_csv = OUT / "scores.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            done = {r["row_id"] for r in csv.DictReader(f)}
    todo = [r for r in pairs if r["row_id"] not in done]
    gens = list(H["gens"])
    cols = PAIR_COLS + ["s2_seen"] + [f"s2_lo_{g}" for g in gens] + ["s3", "s4_score", "s4_same", "s4_method",
                                                                      "s4_inliers"]
    new = not out_csv.exists()
    fh = open(out_csv, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=cols)
    if new:
        w.writeheader()
    trufor_src = Path(os.path.expanduser(a.trufor))
    ref_cache: dict[str, Image.Image] = {}
    t0 = time.time()
    print(f"{len(done)} rows already scored, {len(todo)} to go", flush=True)
    for i in range(0, len(todo), a.chunk):
        chunk = todo[i:i + a.chunk]
        evs, ok = {}, []
        for r in chunk:
            try:
                evs[r["row_id"]] = canonicalise(Image.open(io.BytesIO(make_variant(r["ev_path"], r["variant"]))))
                ok.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {r['row_id']}: {type(e).__name__}", flush=True)
        if not ok:
            continue
        feats = clip_features_batch([evs[r["row_id"]] for r in ok])
        s3 = _trufor(evs, trufor_src) if a.trufor_on else {}
        if a.trufor_on and not any(np.isfinite(list(s3.values()) or [np.nan])):
            print("  WARNING: TruFor returned no scores for this chunk (check --trufor path / output names)", flush=True)
        for r, x in zip(ok, feats):
            ref = None
            if r["ref_path"]:
                if r["ref_path"] not in ref_cache:
                    if len(ref_cache) > 300:
                        ref_cache.clear()
                    ref_cache[r["ref_path"]] = canonicalise(Image.open(r["ref_path"]))
                ref = ref_cache[r["ref_path"]]
            s4 = s4_reference.run(evs[r["row_id"]], ref)
            row = dict(r)
            row.update(_s2_scores(x, int(r["fold"]), H))
            row.update({"s3": s3.get(r["row_id"], float("nan")), "s4_score": s4.score,
                        "s4_same": s4.artifacts.get("same_item"), "s4_method": s4.artifacts.get("method"),
                        "s4_inliers": s4.artifacts.get("inliers")})
            w.writerow(row)
        fh.flush()
        n_done = i + len(chunk)
        rate = (time.time() - t0) / n_done
        print(f"  {n_done}/{len(todo)}  {rate:.2f}s/row  ETA {rate * (len(todo) - n_done) / 60:.0f} min", flush=True)
    fh.close()
    print(f"done -> {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "heads", "score"])
    ap.add_argument("--products", type=int, default=100)
    ap.add_argument("--negatives", type=int, default=100)
    ap.add_argument("--wrong", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--trufor", default="~/vendor/TruFor/test_docker/src")
    ap.add_argument("--no-trufor", dest="trufor_on", action="store_false")
    a = ap.parse_args()
    {"select": select, "heads": heads, "score": score}[a.step](a)


if __name__ == "__main__":
    main()
