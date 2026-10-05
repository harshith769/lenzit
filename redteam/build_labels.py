"""Copy teammates' raw uploads into the official bench layout and write bench/labels.csv.

Input:  data/raw/<item_id>/{item.txt, ref/, undamaged/, damaged/, fake_G1..fake_G4/}
Output: data/lenzit-bench/<item_id>/<image_id><ext>  +  bench/labels.csv (one row per image)

image_id = {item_id}_{role}_{label}_{generator}_{variant}_{nn}
Images are copied with shutil.copy2 only (never re-encoded) so EXIF / C2PA bytes survive.
All validation problems are collected and printed together; if there is any, nothing is written.

Usage (from repo root):
    python -m redteam.build_labels --raw data/raw --out data/lenzit-bench --labels bench/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

REPO_ROOT = Path(__file__).resolve().parents[1]

ITEM_RE = re.compile(r"^it(0[1-9]|[1-3][0-9]|40)$")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
IGNORED_NAMES = {"item.txt", "desktop.ini", ".ds_store"}
CATEGORIES = {"packaging", "hard_goods", "fabric"}
REQUIRED_KEYS = ("phone", "category", "claim")
VARIANT = "original"

# folder, role, label, generator, capture_order, tier -- in nn order
FOLDERS = [
    ("ref", "reference", "real_undamaged", "none", 1, ""),
    ("undamaged", "evidence", "real_undamaged", "none", 1, ""),
    ("damaged", "evidence", "real_damaged", "none", 4, ""),
    ("fake_G1", "evidence", "fake", "G1", 2, "T1"),
    ("fake_G2", "evidence", "fake", "G2", 2, "T1"),
    ("fake_G3", "evidence", "fake", "G3", 2, "T3"),
    ("fake_G4", "evidence", "fake", "G4", 2, "TODO"),
]
FOLDER_NAMES = [f[0] for f in FOLDERS]
REAL_FOLDERS = {"ref", "undamaged", "damaged"}

COLUMNS = ["image_id", "path", "item_id", "phone", "category", "role", "label", "generator",
           "tier", "variant", "claim_text", "damage_bbox", "capture_order"]

EXIF_MAKE, EXIF_MODEL = 0x010F, 0x0110

GATE_REAL_TOTAL, GATE_REAL_DAMAGED, GATE_G1_G2 = 100, 40, 60


@dataclass
class Planned:
    src: Path
    item_id: str
    folder: str
    nn: int
    meta: dict
    row: dict = field(default_factory=dict)


def is_ignored(name: str) -> bool:
    return name.startswith(".") or name.lower() in IGNORED_NAMES


def slugify(text: str) -> str:
    s = text.strip().strip("\x00").lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def phone_slug(make: str, model: str) -> str:
    make_s, model_s = slugify(make), slugify(model)
    if make_s and model_s and (model_s == make_s or model_s.startswith(make_s + "-")):
        return model_s
    return "-".join(p for p in (make_s, model_s) if p)


def read_item_txt(path: Path) -> dict:
    meta = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip().lower()] = value.strip()
    return meta


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_image(path: Path) -> str | None:
    """Return an error string if the file is empty or Pillow cannot read it. Never saves."""
    if path.stat().st_size == 0:
        return f"empty (0-byte) image: {path}"
    if path.suffix.lower() == ".heic" and not HEIF_OK:
        return None  # cannot decode without pillow_heif; never fail because of HEIC
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception as e:  # noqa: BLE001 -- any decoder failure is a validation error
        return f"unreadable image: {path} ({type(e).__name__})"
    return None


def scan(raw: Path) -> tuple[list[Planned], list[str], list[str]]:
    """Walk raw/, returning (planned images, errors, warnings). Reads only."""
    planned: list[Planned] = []
    errors: list[str] = []
    warns: list[str] = []
    if not raw.is_dir():
        return planned, [f"raw folder not found: {raw}"], warns

    seen_hash: dict[str, Path] = {}
    for item_dir in sorted(raw.iterdir()):
        if is_ignored(item_dir.name):
            continue
        if not item_dir.is_dir():
            warns.append(f"WARN skipped stray file: {item_dir}")
            continue
        item_id = item_dir.name
        if not ITEM_RE.match(item_id):
            errors.append(f"bad item folder name (want it01..it40): {item_dir}")
            continue

        meta: dict = {}
        txt = item_dir / "item.txt"
        if not txt.is_file():
            errors.append(f"{item_id}: item.txt missing")
        else:
            meta = read_item_txt(txt)
            for key in REQUIRED_KEYS:
                if not meta.get(key):
                    errors.append(f"{item_id}: item.txt missing or empty '{key}'")
            cat = meta.get("category")
            if cat and cat not in CATEGORIES:
                errors.append(f"{item_id}: category '{cat}' not one of {sorted(CATEGORIES)}")

        images: list[tuple[str, Path]] = []
        for sub in sorted(item_dir.iterdir()):
            if is_ignored(sub.name):
                continue
            if not sub.is_dir():
                warns.append(f"WARN skipped file outside a sorting folder: {sub}")
                continue
            if sub.name not in FOLDER_NAMES:
                errors.append(f"{item_id}: unknown subfolder '{sub.name}' ({sub})")
                continue
            for f in sorted(sub.iterdir(), key=lambda p: p.name):
                if is_ignored(f.name):
                    continue
                if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                    warns.append(f"WARN skipped non-image: {f}")
                    continue
                images.append((sub.name, f))

        if not images:
            errors.append(f"{item_id}: no images at all")
            continue

        images.sort(key=lambda t: (FOLDER_NAMES.index(t[0]), t[1].name))
        for nn, (folder, f) in enumerate(images, start=1):
            bad = check_image(f)
            if bad:
                errors.append(f"{item_id}: {bad}")
            else:
                digest = sha256(f)
                if digest in seen_hash:
                    errors.append(f"duplicate image (same sha256): {seen_hash[digest]} == {f}")
                else:
                    seen_hash[digest] = f
            planned.append(Planned(src=f, item_id=item_id, folder=folder, nn=nn, meta=meta))
    return planned, errors, warns


def check_out_dir(out: Path, raw: Path, repo_root: Path) -> list[str]:
    """Refuse to rmtree anything but a lenzit-bench folder or a pytest temp dir."""
    out, raw, repo_root = out.resolve(), raw.resolve(), repo_root.resolve()
    errors = []
    tmp = Path(tempfile.gettempdir()).resolve()
    in_pytest_tmp = out.is_relative_to(tmp) and any(
        p.startswith("pytest-of-") for p in out.relative_to(tmp).parts)
    if out.name != "lenzit-bench" and not in_pytest_tmp:
        errors.append(f"refusing to delete --out {out}: folder must be named 'lenzit-bench'")
    if out == repo_root or repo_root.is_relative_to(out):
        errors.append(f"refusing to delete --out {out}: it is the repo root or contains it")
    if out == raw or raw.is_relative_to(out):
        errors.append(f"refusing to delete --out {out}: it is --raw or contains it")
    if out.is_relative_to(raw) and out != raw:
        errors.append(f"--out {out} must not be inside --raw")
    if not out.is_relative_to(repo_root):
        errors.append(f"--out {out} must be inside the repo root {repo_root}")
    return errors


def read_header(template: Path) -> tuple[list[str], list[str]]:
    if not template.is_file():
        return [], [f"labels template not found: {template}"]
    with open(template, newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    header = [h.strip() for h in header]
    if set(header) != set(COLUMNS):
        return header, [f"template columns {header} differ from expected {COLUMNS}"]
    return header, []


def exif_phone(path: Path) -> str:
    """Make/Model slug from EXIF, '' if absent. Image.open + getexif only, never saves."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            make, model = str(exif.get(EXIF_MAKE, "") or ""), str(exif.get(EXIF_MODEL, "") or "")
    except Exception:  # noqa: BLE001
        return ""
    return phone_slug(make, model)


def dest_name(p: Planned) -> str:
    _, role, label, gen, _, _ = FOLDERS[FOLDER_NAMES.index(p.folder)]
    ext = p.src.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return f"{p.item_id}_{role}_{label}_{gen}_{VARIANT}_{p.nn:02d}{ext}"


def build_rows(planned: list[Planned], out: Path, repo_root: Path) -> None:
    heic_warned = False
    exif_counts: dict[str, Counter] = {}  # item_id -> EXIF phone slugs of its real photos
    for p in planned:  # real folders come before fakes in nn order, so counts are complete in time
        folder, role, label, gen, order, tier = FOLDERS[FOLDER_NAMES.index(p.folder)]
        name = dest_name(p)
        phone = slugify(p.meta["phone"])
        counts = exif_counts.setdefault(p.item_id, Counter())
        if folder not in REAL_FOLDERS:
            if counts:  # fakes: most common real-photo EXIF phone (ties -> first seen)
                phone = counts.most_common(1)[0][0]
        else:
            if p.src.suffix.lower() == ".heic" and not HEIF_OK:
                if not heic_warned:
                    print("WARN pillow_heif not installed: HEIC phones taken from item.txt")
                    heic_warned = True
            else:
                slug = exif_phone(p.src)
                if slug:
                    phone = slug
                    counts[slug] += 1
                else:
                    print(f"WARN no EXIF: {p.src} (sent via WhatsApp or edited?)")
        dest = out.resolve() / p.item_id / name
        p.row = {
            "image_id": Path(name).stem,
            "path": dest.relative_to(repo_root.resolve()).as_posix(),
            "item_id": p.item_id,
            "phone": phone,
            "category": p.meta["category"],
            "role": role,
            "label": label,
            "generator": gen,
            "tier": tier,
            "variant": VARIANT,
            "claim_text": p.meta["claim"],
            "damage_bbox": "",
            "capture_order": str(order),
        }


def write_outputs(planned: list[Planned], out: Path, labels: Path, header: list[str]) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for p in planned:
        dest = out / p.item_id / dest_name(p)
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(p.src, dest)
    labels.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted((p.row for p in planned), key=lambda r: (r["item_id"], int(r["image_id"].rsplit("_", 1)[1])))
    with open(labels, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def summarize(planned: list[Planned]) -> list[str]:
    rows = [p.row for p in planned]
    n_und = sum(r["label"] == "real_undamaged" for r in rows)
    n_dmg = sum(r["label"] == "real_damaged" for r in rows)
    gens = {g: sum(r["generator"] == g for r in rows) for g in ("G1", "G2", "G3", "G4")}
    lines = [
        "SUMMARY",
        f"items: {len({r['item_id'] for r in rows})}",
        f"distinct phones: {len({r['phone'] for r in rows})}",
        f"real_undamaged: {n_und}",
        f"real_damaged: {n_dmg}",
    ] + [f"fakes {g}: {n}" for g, n in gens.items()]
    if n_und + n_dmg < GATE_REAL_TOTAL:
        lines.append(f"WARN real total < {GATE_REAL_TOTAL} (have {n_und + n_dmg})")
    if n_dmg < GATE_REAL_DAMAGED:
        lines.append(f"WARN real_damaged < {GATE_REAL_DAMAGED} (have {n_dmg})")
    if gens["G1"] + gens["G2"] < GATE_G1_G2:
        lines.append(f"WARN G1+G2 fakes < {GATE_G1_G2} (have {gens['G1'] + gens['G2']})")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/lenzit-bench"))
    ap.add_argument("--labels", type=Path, default=Path("bench/labels.csv"))
    ap.add_argument("--template", type=Path, default=Path("bench/labels_template.csv"))
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args(argv)

    header, errors = read_header(args.template)
    errors += check_out_dir(args.out, args.raw, args.repo_root)
    planned, scan_errors, warns = scan(args.raw)
    errors += scan_errors
    for w in warns:
        print(w)
    if errors:
        for e in errors:
            print(f"ERROR {e}", file=sys.stderr)
        print(f"{len(errors)} error(s); nothing written.", file=sys.stderr)
        return 1

    build_rows(planned, args.out, args.repo_root)
    write_outputs(planned, args.out, args.labels, header)
    for line in summarize(planned):
        print(line)
    print(f"wrote {len(planned)} images to {args.out} and {args.labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
