"""S1 — provenance signal.

run(evidence_bytes, delivery_date) -> SignalResult(name="provenance", ...)

S1 does not decide verdicts; pipeline/run.py does, from these artifacts:
  c2pa_present, c2pa_ai, c2pa_ai_edited, exif_before_delivery   (always present, bool)
  c2pa_claim_generator, exif_datetime, camera                    (optional)

Score (0 = looks genuine, 1 = looks manipulated), first match wins:
  0.95 c2pa_ai | 0.8 c2pa_ai_edited | 0.6 exif_before_delivery | 0.1 valid camera metadata | 0.3 no metadata

Absence of metadata is not evidence: screenshots and messaging apps strip it.
run() never raises; on any parse error it returns score 0.3 with a reason.
"""
from __future__ import annotations

import io
import json
import time
from datetime import date, datetime

from PIL import Image

from pipeline.base import SignalResult

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover
    pass

NAME = "provenance"

# IPTC digital source types written into C2PA manifests (OpenAI, Adobe Firefly, Google, ...)
_AI_EDITED_TYPES = ("compositewithtrainedalgorithmicmedia", "compositesynthetic")
_AI_GENERATED_TYPES = ("trainedalgorithmicmedia", "algorithmicmedia")

_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_DATETIME = 306
_TAG_MAKE = 271
_TAG_MODEL = 272


def _sniff_mime(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp":
        return "image/heic"
    return "application/octet-stream"


def _read_c2pa(data: bytes) -> dict:
    """Returns {present, ai, ai_edited, generator}. Missing manifest is the normal case, not an error."""
    out = {"present": False, "ai": False, "ai_edited": False, "generator": None}
    try:
        import c2pa

        with c2pa.Reader(_sniff_mime(data), io.BytesIO(data)) as reader:
            raw = reader.json()
    except Exception:
        return out  # no manifest, or unreadable manifest -> treat as absent
    if not raw:
        return out

    manifest_store = json.loads(raw)
    out["present"] = True

    active = manifest_store.get("manifests", {}).get(manifest_store.get("active_manifest", ""), {})
    gen = active.get("claim_generator")
    if not gen:
        info = active.get("claim_generator_info") or []
        if isinstance(info, list) and info and isinstance(info[0], dict):
            gen = " ".join(str(info[0].get(k, "")) for k in ("name", "version")).strip() or None
    out["generator"] = gen

    blob = json.dumps(manifest_store).lower().replace("_", "")
    out["ai_edited"] = any(t in blob for t in _AI_EDITED_TYPES)
    for t in _AI_EDITED_TYPES:  # don't let "composite...trainedAlgorithmicMedia" count as fully generated
        blob = blob.replace(t, "")
    out["ai"] = any(t in blob for t in _AI_GENERATED_TYPES)
    return out


def _parse_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, bytes):
        value = value.decode(errors="ignore")
    if isinstance(value, str):
        s = value.strip().rstrip("\x00").replace("Z", "")
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt)
            except ValueError:
                continue
    return None


def _read_exif(data: bytes) -> tuple[datetime | None, str | None]:
    """Returns (capture_time, camera)."""
    with Image.open(io.BytesIO(data)) as img:
        exif = img.getexif()
        sub = exif.get_ifd(_EXIF_IFD)
        raw_dt = sub.get(_TAG_DATETIME_ORIGINAL) or sub.get(_TAG_DATETIME_DIGITIZED) or exif.get(_TAG_DATETIME)
        make = str(exif.get(_TAG_MAKE) or "").strip().rstrip("\x00")
        model = str(exif.get(_TAG_MODEL) or "").strip().rstrip("\x00")
    camera = " ".join(p for p in (make, model) if p) or None
    return _parse_dt(raw_dt), camera


def run(evidence_bytes: bytes, delivery_date: date | None) -> SignalResult:
    t0 = time.perf_counter()
    artifacts = {
        "c2pa_present": False,
        "c2pa_ai": False,
        "c2pa_ai_edited": False,
        "exif_before_delivery": False,
    }

    def _done(score: float, reason: str) -> SignalResult:
        return SignalResult(
            name=NAME,
            score=score,
            reason=reason,
            artifacts=artifacts,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    try:
        c2 = _read_c2pa(evidence_bytes)
        artifacts["c2pa_present"] = c2["present"]
        artifacts["c2pa_ai"] = c2["ai"]
        artifacts["c2pa_ai_edited"] = c2["ai_edited"]
        if c2["generator"]:
            artifacts["c2pa_claim_generator"] = c2["generator"]

        captured, camera = _read_exif(evidence_bytes)
        if captured:
            artifacts["exif_datetime"] = captured.isoformat()
        if camera:
            artifacts["camera"] = camera

        delivered = _parse_dt(delivery_date)
        if captured and delivered and captured.date() < delivered.date():
            artifacts["exif_before_delivery"] = True
    except Exception:
        return _done(0.3, "We couldn't read this photo's metadata, so provenance could not be checked.")

    if artifacts["c2pa_ai"]:
        return _done(0.95, "This image carries Content Credentials saying it was generated by an AI tool.")
    if artifacts["c2pa_ai_edited"]:
        return _done(0.8, "This image carries Content Credentials saying it was edited with an AI tool.")
    if artifacts["exif_before_delivery"]:
        return _done(0.6, "The photo's capture date is earlier than the delivery date.")
    if captured or camera:
        return _done(0.1, "The photo has normal camera metadata with no signs of AI generation.")
    return _done(0.3, "The photo has no metadata, which is common for screenshots and shared images.")
