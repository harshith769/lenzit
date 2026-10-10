import io
from datetime import date

import pytest
from PIL import Image

from pipeline import s1_provenance
from pipeline.base import SignalResult
from security.upload_guard import UploadRejected, validate_upload

REQUIRED = {"c2pa_present", "c2pa_ai", "c2pa_ai_edited", "exif_before_delivery"}


def _jpeg(w=64, h=64, exif_dt=None, camera=None) -> bytes:
    img = Image.new("RGB", (w, h), (120, 80, 40))
    exif = Image.Exif()
    if exif_dt:
        exif.get_ifd(0x8769)[36867] = exif_dt
    if camera:
        exif[271], exif[272] = camera
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes())
    return buf.getvalue()


# ---------- upload guard ----------
def test_valid_jpeg_returns_image():
    img = validate_upload(_jpeg())
    assert isinstance(img, Image.Image) and img.size == (64, 64)


def test_valid_png_returns_image():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, "PNG")
    assert validate_upload(buf.getvalue()).size == (10, 10)


def test_text_renamed_jpg_rejected_422():
    with pytest.raises(UploadRejected) as e:
        validate_upload(b"hello this is a text file pretending to be a jpg")
    assert e.value.status_code == 422


def test_fake_magic_bytes_rejected_422():
    with pytest.raises(UploadRejected) as e:
        validate_upload(b"\xff\xd8\xff" + b"garbage" * 100)
    assert e.value.status_code == 422


def test_oversize_rejected_413():
    with pytest.raises(UploadRejected) as e:
        validate_upload(b"\xff\xd8\xff" + b"\x00" * 12_000_000)
    assert e.value.status_code == 413


def test_custom_max_bytes():
    with pytest.raises(UploadRejected) as e:
        validate_upload(_jpeg(), max_bytes=100)
    assert e.value.status_code == 413


def test_decompression_bomb_rejected_413():
    buf = io.BytesIO()
    Image.new("1", (8000, 7000)).save(buf, "PNG")  # 56 MP, tiny on disk
    with pytest.raises(UploadRejected) as e:
        validate_upload(buf.getvalue())
    assert e.value.status_code == 413


def test_empty_rejected():
    with pytest.raises(UploadRejected):
        validate_upload(b"")


# ---------- S1 provenance ----------
def test_s1_contract_shape():
    r = s1_provenance.run(_jpeg(), date(2026, 10, 1))
    assert isinstance(r, SignalResult)
    assert r.name == "provenance"
    assert REQUIRED <= set(r.artifacts)
    assert all(isinstance(r.artifacts[k], bool) for k in REQUIRED)
    assert isinstance(r.reason, str) and r.reason
    assert r.latency_ms >= 0


def test_s1_no_metadata_scores_0_3():
    r = s1_provenance.run(_jpeg(), date(2026, 10, 1))
    assert r.score == 0.3


def test_s1_camera_metadata_scores_0_1():
    r = s1_provenance.run(_jpeg(exif_dt="2026:10:05 10:00:00", camera=("Apple", "iPhone 14")), date(2026, 10, 1))
    assert r.score == 0.1
    assert r.artifacts["camera"] == "Apple iPhone 14"
    assert r.artifacts["exif_before_delivery"] is False


def test_s1_exif_before_delivery_scores_0_6():
    r = s1_provenance.run(_jpeg(exif_dt="2026:09:01 10:00:00"), date(2026, 10, 1))
    assert r.artifacts["exif_before_delivery"] is True
    assert r.score == 0.6


def test_s1_delivery_date_none():
    r = s1_provenance.run(_jpeg(exif_dt="2026:09:01 10:00:00"), None)
    assert r.artifacts["exif_before_delivery"] is False
    assert r.score == 0.1


def test_s1_garbage_never_raises():
    r = s1_provenance.run(b"not an image", date(2026, 10, 1))
    assert r.score == 0.3
    assert REQUIRED <= set(r.artifacts)


def test_s1_c2pa_detection_logic(monkeypatch):
    monkeypatch.setattr(s1_provenance, "_read_c2pa", lambda d: {"present": True, "ai": True, "ai_edited": False, "generator": "ChatGPT"})
    r = s1_provenance.run(_jpeg(), date(2026, 10, 1))
    assert r.score == 0.95 and r.artifacts["c2pa_ai"] is True
    assert r.artifacts["c2pa_claim_generator"] == "ChatGPT"


def test_s1_c2pa_edited_scores_0_8(monkeypatch):
    monkeypatch.setattr(s1_provenance, "_read_c2pa", lambda d: {"present": True, "ai": False, "ai_edited": True, "generator": None})
    r = s1_provenance.run(_jpeg(), date(2026, 10, 1))
    assert r.score == 0.8
