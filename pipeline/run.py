"""Single integration point: the API calls analyse(); everything else in pipeline/ is internal.

Real signals so far: S2 global_synthetic (set LENZIT_S2_MODE=off to use its stub, e.g. in light tests).
Still stubs: provenance (S1), local_edit (S3), reference (S4), plausibility (S5); fusion is a plain
average until pipeline/fusion.py lands (Oct 7). The returned keys are the frozen contract.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import time

from PIL import Image

from . import s2_global
from .canonical import canonicalise
from .stubs import run_stub

MODEL_VERSION = "lz-2026.10.05-s2v1"
SIGNALS = ["provenance", "global_synthetic", "local_edit", "reference", "plausibility"]


def _signals(ev, ref, claim):
    out = []
    for name in SIGNALS:
        if name == "global_synthetic" and os.environ.get("LENZIT_S2_MODE", "on") == "on":
            out.append(s2_global.run(ev, ref, claim))
        else:
            out.append(run_stub(name, ev, ref, claim))
    return out


def analyse(evidence: bytes, reference: bytes | None, claim: str,
            delivery_date: dt.date | None = None) -> dict:
    t0 = time.time()
    ev = canonicalise(Image.open(io.BytesIO(evidence)))
    ref = canonicalise(Image.open(io.BytesIO(reference))) if reference else None
    signals = _signals(ev, ref, claim)
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
