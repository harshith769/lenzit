"""Build bench/labels_fraudbench.csv from a local copy of FraudBench (CC BY-NC-SA 4.0, arXiv 2605.08820).

Layout:  <Category>/Positive/Review_x/*            -> real_undamaged (positive reviews: the sources of the fakes)
         <Category>/Negative/Review_x/*            -> real_damaged (negative reviews)
         <Category>/DeepFake/<generator>/Review_x/* -> fake, edited from Positive/Review_x
         <Category>/DeepFake/Metadata/Edit_x.json  -> claim text per fake
Groups (item_id): a Positive review and all its fakes share one group; each Negative review is its own group.
The CSV is regenerable and gitignored (licence + size).
Usage: python -m bench.fraudbench_labels data/fraudbench bench/labels_fraudbench.csv
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
COLUMNS = ["image_id", "path", "item_id", "phone", "category", "role", "label", "generator",
           "tier", "variant", "claim_text", "damage_bbox", "capture_order"]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def claims_for(cat_dir: Path) -> dict:
    out = {}
    meta_dir = cat_dir / "DeepFake" / "Metadata"
    for meta in sorted(meta_dir.glob("*.json")) if meta_dir.exists() else []:
        d = json.loads(meta.read_text())
        review = d.get("source_review")
        for p in d.get("per_image", []):
            for gen, o in (p.get("outputs") or {}).items():
                if (o or {}).get("status") == "ok":
                    out[(gen, review, p["image"])] = p.get("reviewer_comment") or ""
    return out


def main(root: str, out_csv: str) -> None:
    root_p = Path(root)
    rows, counts, with_claim = [], Counter(), 0
    for cat_dir in sorted(p for p in root_p.iterdir() if p.is_dir() and not p.name.startswith(".")):
        cat = slug(cat_dir.name)
        claims = claims_for(cat_dir)
        for cls, label, tag in (("Positive", "real_undamaged", "pos"), ("Negative", "real_damaged", "neg")):
            for img in sorted((cat_dir / cls).glob("*/*")):
                if img.suffix.lower() not in EXTS:
                    continue
                review = img.parent.name
                rows.append({"image_id": f"fb_{cat}_{tag}_{review}_{img.stem}", "path": str(img),
                             "item_id": f"fb-{cat}-{tag}-{review}", "category": cat, "role": "evidence",
                             "label": label, "generator": "none", "variant": "original"})
                counts[label] += 1
        fake_root = cat_dir / "DeepFake"
        gen_dirs = sorted(p for p in fake_root.iterdir() if p.is_dir() and p.name != "Metadata") if fake_root.exists() else []
        for gen_dir in gen_dirs:
            gen = gen_dir.name
            for img in sorted(gen_dir.glob("*/*")):
                if img.suffix.lower() not in EXTS:
                    continue
                review = img.parent.name
                claim = claims.get((gen, review, img.name), "")
                with_claim += bool(claim)
                rows.append({"image_id": f"fb_{cat}_{slug(gen)}_{review}_{img.stem}", "path": str(img),
                             "item_id": f"fb-{cat}-pos-{review}", "category": cat, "role": "evidence",
                             "label": "fake", "generator": gen, "variant": "original", "claim_text": claim})
                counts[f"fake {gen}"] += 1
    ids = [r["image_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate image_id"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    for k, v in sorted(counts.items()):
        print(f"  {k:32s} {v}")
    n_fake = sum(v for k, v in counts.items() if k.startswith("fake"))
    print(f"total {len(rows)} images | {len({r['item_id'] for r in rows})} groups | "
          f"{with_claim}/{n_fake} fakes with claim text -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
