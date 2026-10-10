"""Single integration point: the API calls analyse(); everything else in pipeline/ is internal.

Decision policy v1 (models/decision_v1.json, measured on Lenzit-Bench v1, Oct 10):
  * risk_score = S2 global_synthetic score; S2 >= t_likely_manipulated -> likely_manipulated,
    S2 >= t_needs_verification -> needs_verification, else likely_genuine.
  * S4 reference: same-item similarity below s4_wrong_item_same_below -> at least needs_verification.
  * S1 provenance (when pipeline/s1_provenance.py exists): C2PA "AI-generated" -> likely_manipulated (hard rule);
    C2PA "AI-edited" or EXIF capture before delivery -> at least needs_verification.
  * S3 local_edit (TruFor, LENZIT_S3_MODE=local) is shown as a heatmap only; it never moves the verdict
    (it scored FraudBench's full-frame fakes below real photos, AUROC 0.39).
  * Signals that could not run keep score 0.5, say why, and are listed in unavailable_signals.
The returned keys are the frozen contract.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import time
from pathlib import Path

from PIL import Image

from . import s2_global, s4_reference
from .base import SignalResult
from .canonical import canonicalise
from .heatmap import overlay_png

MODEL_VERSION = "lz-2026.10.10-s2v1-s4v1-dec1"
SIGNALS = ["provenance", "global_synthetic", "local_edit", "reference", "plausibility"]
POLICY = json.loads((Path(__file__).resolve().parents[1] / "models" / "decision_v1.json").read_text())
RANK = {"likely_genuine": 0, "needs_verification": 1, "likely_manipulated": 2}


def _unavailable(name: str, why: str) -> SignalResult:
    return SignalResult(name, 0.5, why, {"available": False})


# Fallback until Sriman's pipeline/s1_provenance.py is merged: look for an embedded C2PA manifest and its IPTC
# digitalSourceType. This only READS the declaration; it does not verify the signature (s1_provenance does).
_AI_FULL = b"digitalsourcetype/trainedAlgorithmicMedia"
_AI_EDIT = (b"digitalsourcetype/compositeWithTrainedAlgorithmicMedia", b"digitalsourcetype/algorithmicMedia",
            b"compositeSynthetic")


def _c2pa_scan(data: bytes) -> SignalResult:
    t0 = time.time()
    has = b"c2pa" in data
    full = has and _AI_FULL in data
    edited = has and not full and any(m in data for m in _AI_EDIT)
    art = {"c2pa_present": has, "c2pa_ai": full, "c2pa_ai_edited": edited, "signature_verified": False}
    if full:
        return SignalResult("provenance", 0.95, "Content Credentials in the file declare the image AI-generated", art,
                            int((time.time() - t0) * 1000))
    if edited:
        return SignalResult("provenance", 0.8, "Content Credentials in the file declare AI editing", art,
                            int((time.time() - t0) * 1000))
    if has:
        return SignalResult("provenance", 0.3, "Content Credentials present, with no AI declaration found", art,
                            int((time.time() - t0) * 1000))
    return SignalResult("provenance", 0.3, "No Content Credentials in the file (common: most apps strip them)", art,
                        int((time.time() - t0) * 1000))


def _s1(evidence_bytes: bytes, delivery_date):
    try:
        from . import s1_provenance  # Sriman's module; optional until merged
    except ImportError:
        return _c2pa_scan(evidence_bytes)
    try:
        return s1_provenance.run(evidence_bytes, delivery_date)
    except Exception as e:  # noqa: BLE001 -- a bonus signal must never break a claim
        return _unavailable("provenance", f"Provenance check failed ({type(e).__name__})")


def _s3(ev: Image.Image):
    if os.environ.get("LENZIT_S3_MODE", "off") != "local":
        return _unavailable("local_edit", "Local-edit heatmap (TruFor) is not enabled on this server")
    try:
        from . import s3_local
        return s3_local.run(ev)
    except Exception as e:  # noqa: BLE001
        return _unavailable("local_edit", f"Local-edit check failed ({type(e).__name__})")


def _signals(evidence_bytes, ev, ref, claim, delivery_date) -> dict[str, SignalResult]:
    on = lambda k: os.environ.get(k, "on") == "on"  # noqa: E731
    out = {
        "provenance": _s1(evidence_bytes, delivery_date),
        "global_synthetic": s2_global.run(ev, ref, claim) if on("LENZIT_S2_MODE")
        else _unavailable("global_synthetic", "AI-image detector is switched off"),
        "local_edit": _s3(ev),
        "reference": s4_reference.run(ev, ref, claim) if on("LENZIT_S4_MODE")
        else _unavailable("reference", "Reference check is switched off"),
        "plausibility": _unavailable("plausibility", "Plausibility check is not part of this version"),
    }
    if ref is None and out["reference"].artifacts.get("available", True):
        out["reference"].artifacts["available"] = False      # no reference uploaded
    return out


def _decide(sig: dict[str, SignalResult]) -> tuple[str, float, list[str]]:
    why = []
    s2 = sig["global_synthetic"]
    risk = float(s2.score)
    if not s2.artifacts.get("available", True):
        verdict = "needs_verification"
        why.append("AI-image detector unavailable")
    elif risk >= POLICY["t_likely_manipulated"]:
        verdict = "likely_manipulated"
    elif risk >= POLICY["t_needs_verification"]:
        verdict = "needs_verification"
    else:
        verdict = "likely_genuine"

    def floor(v, reason):
        nonlocal verdict
        if RANK[v] > RANK[verdict]:
            verdict = v
        why.append(reason)

    same = sig["reference"].artifacts.get("same_item")
    if same is not None and same < POLICY["s4_wrong_item_same_below"]:
        floor("needs_verification", "photo may not show the seller's item")
    a1 = sig["provenance"].artifacts
    if a1.get("c2pa_ai"):
        floor("likely_manipulated", "Content Credentials declare AI generation")
    elif a1.get("c2pa_ai_edited"):
        floor("needs_verification", "Content Credentials declare AI editing")
    if a1.get("exif_before_delivery"):
        floor("needs_verification", "photo was taken before the delivery date")
    return verdict, round(min(max(risk, 0.0), 1.0), 3), why


def analyse(evidence: bytes, reference: bytes | None, claim: str,
            delivery_date: dt.date | None = None) -> dict:
    t0 = time.time()
    ev = canonicalise(Image.open(io.BytesIO(evidence)))
    ref = canonicalise(Image.open(io.BytesIO(reference))) if reference else None
    sig = _signals(evidence, ev, ref, claim, delivery_date)
    verdict, risk, rules = _decide(sig)

    mask = sig["local_edit"].artifacts.get("heatmap")
    if mask is None:
        mask = sig["reference"].artifacts.get("change_mask") if ref is not None else None
    heat = overlay_png(ev, mask) if mask is not None else None

    return {
        "verdict": verdict,
        "risk_score": risk,
        "signals": {n: {"score": round(float(s.score), 4), "reason": s.reason} for n, s in sig.items()},
        "unavailable_signals": [n for n, s in sig.items() if not s.artifacts.get("available", True)],
        "rules_applied": rules,
        "next_step": "refund_as_normal" if verdict == "likely_genuine" else "send_recapture_link",
        "model_version": MODEL_VERSION,
        "latency_ms": int((time.time() - t0) * 1000),
        "artifacts": {"heatmap_png": heat},
    }
