"""Tests for redteam/build_labels.py. All images are tiny and made here in tmp_path."""
from __future__ import annotations

import csv
import hashlib
import itertools
import shutil
from pathlib import Path

import pytest
from PIL import Image

from redteam import build_labels as bl

TEMPLATE_HEADER = ("image_id,path,item_id,phone,category,role,label,generator,tier,variant,"
                   "claim_text,damage_bbox,capture_order\n")
_colors = itertools.count(1)


def make_image(path: Path, make: str | None = None, model: str | None = None) -> Path:
    """Write a tiny image with a unique colour (so sha256 never collides by accident)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = next(_colors)
    img = Image.new("RGB", (8, 8), (c % 256, (c // 256) % 256, 7))
    fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    kw = {} if fmt == "PNG" else {"comment": f"img{c}"}  # tiny JPEGs can otherwise encode identically
    if make or model:
        exif = Image.Exif()
        if make:
            exif[bl.EXIF_MAKE] = make
        if model:
            exif[bl.EXIF_MODEL] = model
        img.save(path, fmt, exif=exif.tobytes(), **kw)
    else:
        img.save(path, fmt, **kw)
    return path


def write_item_txt(item_dir: Path, phone="pixel-7", category="packaging",
                   claim="box arrived crushed") -> None:
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "item.txt").write_text(f"phone={phone}\ncategory={category}\nclaim={claim}\n")


@pytest.fixture
def env(tmp_path):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    out = tmp_path / "data" / "lenzit-bench"
    labels = tmp_path / "bench" / "labels.csv"
    template = tmp_path / "bench" / "labels_template.csv"
    template.parent.mkdir(parents=True)
    template.write_text(TEMPLATE_HEADER)

    def run(*extra):
        argv = ["--raw", str(raw), "--out", str(out), "--labels", str(labels),
                "--template", str(template), "--repo-root", str(tmp_path), *extra]
        return bl.main(argv)

    return {"root": tmp_path, "raw": raw, "out": out, "labels": labels, "run": run}


def read_rows(labels: Path) -> list[dict]:
    with open(labels, newline="") as f:
        return list(csv.DictReader(f))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_full_item(raw: Path) -> Path:
    item = raw / "it07"
    write_item_txt(item, claim="lid dented, corner torn")
    make_image(item / "ref" / "r.jpg", "samsung", "SM-A546E")
    make_image(item / "undamaged" / "u.png", "samsung", "SM-A546E")
    make_image(item / "damaged" / "b.jpg", "samsung", "SM-A546E")
    make_image(item / "damaged" / "a.JPEG", "samsung", "SM-A546E")
    for g in ("G1", "G2", "G3", "G4"):
        make_image(item / f"fake_{g}" / "x.png")
    (item / "damaged" / "desktop.ini").write_text("junk")
    (item / "ref" / ".DS_Store").write_text("junk")
    return item


# 1. one full item ----------------------------------------------------------------------------
def test_full_item(env):
    build_full_item(env["raw"])
    assert env["run"]() == 0
    rows = read_rows(env["labels"])
    assert list(rows[0].keys()) == TEMPLATE_HEADER.strip().split(",")
    expected = [
        ("it07_reference_real_undamaged_none_original_01", ".jpg", "reference", "real_undamaged", "none", "", "1"),
        ("it07_evidence_real_undamaged_none_original_02", ".png", "evidence", "real_undamaged", "none", "", "1"),
        ("it07_evidence_real_damaged_none_original_03", ".jpg", "evidence", "real_damaged", "none", "", "4"),
        ("it07_evidence_real_damaged_none_original_04", ".jpg", "evidence", "real_damaged", "none", "", "4"),
        ("it07_evidence_fake_G1_original_05", ".png", "evidence", "fake", "G1", "T1", "2"),
        ("it07_evidence_fake_G2_original_06", ".png", "evidence", "fake", "G2", "T1", "2"),
        ("it07_evidence_fake_G3_original_07", ".png", "evidence", "fake", "G3", "T3", "2"),
        ("it07_evidence_fake_G4_original_08", ".png", "evidence", "fake", "G4", "TODO", "2"),
    ]
    assert len(rows) == len(expected)
    for row, (iid, ext, role, label, gen, tier, order) in zip(rows, expected):
        assert row["image_id"] == iid
        assert row["path"] == f"data/lenzit-bench/it07/{iid}{ext}"
        assert (env["root"] / row["path"]).is_file()
        assert (row["role"], row["label"], row["generator"]) == (role, label, gen)
        assert (row["tier"], row["capture_order"]) == (tier, order)
        assert row["variant"] == "original"
        assert row["item_id"] == "it07" and row["category"] == "packaging"
        assert row["claim_text"] == "lid dented, corner torn"
        assert row["damage_bbox"] == ""
    # inside a folder, files are sorted by name: a.JPEG (03) before b.jpg (04)
    assert sha(env["out"] / "it07" / "it07_evidence_real_damaged_none_original_03.jpg") == \
        sha(env["raw"] / "it07" / "damaged" / "a.JPEG")
    assert sorted(p.name for p in (env["out"] / "it07").iterdir()) == \
        sorted(iid + ext for iid, ext, *_ in expected)


# 2. dented-box item with only damaged/ --------------------------------------------------------
def test_only_damaged_item_is_valid(env):
    item = env["raw"] / "it03"
    write_item_txt(item, phone="iphone-13")
    make_image(item / "damaged" / "box1.jpg")
    make_image(item / "damaged" / "box2.jpg")
    assert env["run"]() == 0
    rows = read_rows(env["labels"])
    assert [r["image_id"] for r in rows] == [
        "it03_evidence_real_damaged_none_original_01",
        "it03_evidence_real_damaged_none_original_02",
    ]


# 3. rerun is byte-identical --------------------------------------------------------------------
def test_rerun_identical(env):
    build_full_item(env["raw"])
    assert env["run"]() == 0
    labels1 = env["labels"].read_bytes()
    files1 = {p.relative_to(env["out"]): sha(p) for p in env["out"].rglob("*") if p.is_file()}
    (env["out"] / "it07" / "stale_leftover.jpg").write_bytes(b"old")  # must disappear on rerun
    assert env["run"]() == 0
    files2 = {p.relative_to(env["out"]): sha(p) for p in env["out"].rglob("*") if p.is_file()}
    assert env["labels"].read_bytes() == labels1
    assert files2 == files1


# 4. copied bytes identical to source ----------------------------------------------------------
def test_copy_preserves_bytes(env):
    item = build_full_item(env["raw"])
    assert env["run"]() == 0
    src_hashes = sorted(sha(p) for p in item.rglob("*")
                        if p.is_file() and p.suffix.lower() in bl.IMAGE_EXTS)
    dst_hashes = sorted(sha(p) for p in env["out"].rglob("*") if p.is_file())
    assert src_hashes == dst_hashes
    ref = env["out"] / "it07" / "it07_reference_real_undamaged_none_original_01.jpg"
    with Image.open(ref) as im:
        assert im.getexif().get(bl.EXIF_MODEL) == "SM-A546E"


# 5. EXIF phone slug + fallback -----------------------------------------------------------------
def test_exif_phone_and_fallback(env, capsys):
    item = env["raw"] / "it01"
    write_item_txt(item, phone="Pixel 7")
    make_image(item / "ref" / "r.jpg", "samsung", "SM-A546E")
    make_image(item / "undamaged" / "u.jpg", "Google", "Google Pixel_7  Pro!")
    no_exif = make_image(item / "damaged" / "whatsapp.jpg")
    make_image(item / "fake_G1" / "f.jpg", "Apple", "iPhone 13")  # fakes' own EXIF is ignored
    assert env["run"]() == 0
    phones = [r["phone"] for r in read_rows(env["labels"])]
    # tie 1-1 between real EXIF phones -> first seen (ref) wins for the fake
    assert phones == ["samsung-sm-a546e", "google-pixel-7-pro", "pixel-7", "samsung-sm-a546e"]
    out = capsys.readouterr().out
    assert f"WARN no EXIF: {no_exif} (sent via WhatsApp or edited?)" in out
    assert out.count("WARN no EXIF") == 1


def test_fake_phone_is_most_common_real_exif_phone(env):
    item = env["raw"] / "it02"
    write_item_txt(item, phone="Pixel 7")
    make_image(item / "ref" / "r.jpg", "Google", "Pixel 7")
    make_image(item / "damaged" / "d1.jpg", "samsung", "SM-A546E")
    make_image(item / "damaged" / "d2.jpg", "samsung", "SM-A546E")
    make_image(item / "fake_G2" / "f.png")
    only_fakes = env["raw"] / "it03"  # no real photo with EXIF -> item.txt phone
    write_item_txt(only_fakes, phone="Moto G54")
    make_image(only_fakes / "undamaged" / "whatsapp.jpg")
    make_image(only_fakes / "fake_G1" / "f.png")
    assert env["run"]() == 0
    phones = {r["image_id"]: r["phone"] for r in read_rows(env["labels"])}
    assert phones["it02_evidence_fake_G2_original_04"] == "samsung-sm-a546e"
    assert phones["it03_evidence_fake_G1_original_02"] == "moto-g54"


def test_phone_slug():
    assert bl.phone_slug("samsung", "SM-A546E") == "samsung-sm-a546e"
    assert bl.phone_slug("Apple", "iPhone 13") == "apple-iphone-13"
    assert bl.phone_slug("Google", "Google Pixel 7") == "google-pixel-7"
    assert bl.phone_slug("OnePlus\x00", "CPH__2451") == "oneplus-cph-2451"


# 6. validation errors --------------------------------------------------------------------------
@pytest.mark.parametrize("setup, message", [
    (lambda raw: make_image(raw / "item07" / "ref" / "a.jpg"), "bad item folder name"),
    (lambda raw: make_image(raw / "it41" / "ref" / "a.jpg"), "bad item folder name"),
    (lambda raw: make_image(raw / "it02" / "ref" / "a.jpg"), "item.txt missing"),
    (lambda raw: (write_item_txt(raw / "it02", claim=""), make_image(raw / "it02" / "ref" / "a.jpg")),
     "missing or empty 'claim'"),
    (lambda raw: ((raw / "it02").mkdir(), (raw / "it02" / "item.txt").write_text("category=fabric\nclaim=x\n"),
                  make_image(raw / "it02" / "ref" / "a.jpg")), "missing or empty 'phone'"),
    (lambda raw: (write_item_txt(raw / "it02", category="shoes"), make_image(raw / "it02" / "ref" / "a.jpg")),
     "category 'shoes'"),
    (lambda raw: (write_item_txt(raw / "it02"), make_image(raw / "it02" / "fakes" / "a.jpg")),
     "unknown subfolder 'fakes'"),
    (lambda raw: write_item_txt(raw / "it02"), "no images at all"),
])
def test_each_validation_error(env, capsys, setup, message):
    setup(env["raw"])
    assert env["run"]() == 1
    assert message in capsys.readouterr().err
    assert not env["out"].exists() and not env["labels"].exists()


def test_duplicate_sha_reports_both_paths(env, capsys):
    write_item_txt(env["raw"] / "it01")
    write_item_txt(env["raw"] / "it02")
    a = make_image(env["raw"] / "it01" / "ref" / "a.jpg")
    b = env["raw"] / "it02" / "damaged" / "copy.jpg"
    b.parent.mkdir(parents=True)
    shutil.copy(a, b)
    assert env["run"]() == 1
    err = capsys.readouterr().err
    assert "duplicate image" in err and str(a) in err and str(b) in err


def test_unreadable_and_empty_images(env, capsys):
    item = env["raw"] / "it05"
    write_item_txt(item)
    make_image(item / "ref" / "ok.jpg")
    (item / "damaged").mkdir()
    empty = item / "damaged" / "empty.jpg"
    empty.write_bytes(b"")
    garbage = item / "damaged" / "garbage.png"
    garbage.write_bytes(b"this is not a png at all")
    assert env["run"]() == 1
    err = capsys.readouterr().err
    assert f"empty (0-byte) image: {empty}" in err
    assert f"unreadable image: {garbage}" in err
    assert not env["out"].exists()


def test_several_errors_in_one_run(env, capsys):
    make_image(env["raw"] / "bad_name" / "ref" / "a.jpg")
    make_image(env["raw"] / "it02" / "ref" / "a.jpg")                  # no item.txt
    write_item_txt(env["raw"] / "it03", category="toys")
    make_image(env["raw"] / "it03" / "misc" / "a.jpg")                 # bad category + subfolder
    write_item_txt(env["raw"] / "it04")                                # no images
    make_image(env["raw"] / "it05" / "ref" / "ok.jpg")                 # would be fine...
    write_item_txt(env["raw"] / "it05")
    assert env["run"]() == 1
    err = capsys.readouterr().err
    for msg in ("bad item folder name", "it02: item.txt missing", "category 'toys'",
                "unknown subfolder 'misc'", "it04: no images at all"):
        assert msg in err
    assert err.count("ERROR ") == 6  # it03 also has no images left after the bad subfolder
    assert not env["out"].exists() and not env["labels"].exists()


# rmtree guard ----------------------------------------------------------------------------------
def test_out_dir_guard(env, capsys, monkeypatch):
    write_item_txt(env["raw"] / "it01")
    make_image(env["raw"] / "it01" / "ref" / "a.jpg")
    sentinel = env["root"] / "data" / "keep.txt"
    sentinel.write_text("keep")

    def run_out(out: Path) -> str:
        argv = ["--raw", str(env["raw"]), "--out", str(out), "--labels", str(env["labels"]),
                "--template", str(env["root"] / "bench" / "labels_template.csv"),
                "--repo-root", str(env["root"])]
        assert bl.main(argv) == 1
        return capsys.readouterr().err

    assert "repo root" in run_out(env["root"])
    assert "is --raw" in run_out(env["raw"])
    assert "is --raw or contains it" in run_out(env["root"] / "data")  # parent of raw
    assert sentinel.read_text() == "keep" and (env["raw"] / "it01" / "item.txt").exists()

    # Pretend tmp_path is NOT a pytest temp dir: only a folder named lenzit-bench may be deleted.
    monkeypatch.setattr(bl.tempfile, "gettempdir", lambda: str(env["root"] / "elsewhere"))
    other = env["root"] / "data" / "my_stuff"
    other.mkdir()
    (other / "precious.txt").write_text("x")
    assert "must be named 'lenzit-bench'" in run_out(other)
    assert (other / "precious.txt").read_text() == "x"
    assert env["run"]() == 0  # data/lenzit-bench is still allowed


# 7. summary counts and gate WARNs --------------------------------------------------------------
def test_summary_and_gate_warns(env, capsys):
    build_full_item(env["raw"])
    item = env["raw"] / "it08"
    write_item_txt(item, phone="iphone-13")
    make_image(item / "damaged" / "d.jpg")
    make_image(item / "fake_G1" / "f.jpg")
    assert env["run"]() == 0
    out = capsys.readouterr().out
    for line in ("items: 2", "distinct phones: 2", "real_undamaged: 2", "real_damaged: 3",
                 "fakes G1: 2", "fakes G2: 1", "fakes G3: 1", "fakes G4: 1",
                 "WARN real total < 100 (have 5)", "WARN real_damaged < 40 (have 3)",
                 "WARN G1+G2 fakes < 60 (have 3)"):
        assert line in out


def test_no_gate_warns_when_met(env, capsys, monkeypatch):
    monkeypatch.setattr(bl, "GATE_REAL_TOTAL", 2)
    monkeypatch.setattr(bl, "GATE_REAL_DAMAGED", 1)
    monkeypatch.setattr(bl, "GATE_G1_G2", 1)
    build_full_item(env["raw"])
    assert env["run"]() == 0
    assert "WARN" not in capsys.readouterr().out
