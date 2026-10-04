"""Stub signals so the API and frontend can be built before real models land."""
from __future__ import annotations
from typing import Optional
from PIL import Image
from .base import SignalResult

SIGNAL_NAMES = ["provenance", "global_synthetic", "local_edit", "reference", "plausibility"]


def run_stub(name: str, evidence: Image.Image, reference: Optional[Image.Image], claim: str) -> SignalResult:
    return SignalResult(name=name, score=0.5, reason=f"{name}: stub result (model not wired yet)")


def run_all(evidence: Image.Image, reference: Optional[Image.Image], claim: str) -> list[SignalResult]:
    return [run_stub(n, evidence, reference, claim) for n in SIGNAL_NAMES]
