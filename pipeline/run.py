"""Single integration point: the API calls analyse(); everything else in pipeline/ is internal.
STUB until Oct 8 12:00: uses stub signals and an average instead of trained fusion.
The returned keys are the frozen contract (work briefs, section 3)."""
from __future__ import annotations

import datetime as dt
import io
import time

from PIL import Image

from .canonical import canonicalise
from .stubs import run_all

MODEL_VERSION = "lz-stub-2026.10.05"


def analyse(evidence: bytes, reference: bytes | None, claim: str,
            delivery_date: dt.date | None = None) -> dict:
    t0 = time.time()
    ev = canonicalise(Image.open(io.BytesIO(evidence)))
    ref = canonicalise(Image.open(io.BytesIO(reference))) if reference else None
    signals = run_all(ev, ref, claim)
    risk = sum(s.score for s in signals) / len(signals)
    verdict = ("likely_manipulated" if risk >= 0.7
               else "needs_verification" if risk >= 0.4 else "likely_genuine")
    return {
        "verdict": verdict,
        "risk_score": round(risk, 3),
        "signals": {s.name: {"score": s.score, "reason": s.reason} for s in signals},
        "unavailable_signals": [],
        "next_step": "refund_as_normal" if verdict == "likely_genuine" else "send_recapture_link",
        "model_version": MODEL_VERSION,
        "latency_ms": int((time.time() - t0) * 1000),
        "artifacts": {"heatmap_png": None},
    }
