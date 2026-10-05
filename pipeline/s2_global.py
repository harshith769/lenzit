"""S2 global synthetic detector (Harshith).

Frozen CLIP ViT-L/14 features (5 native crops of the canonical image) + a linear head trained on
FraudBench (models/s2_head_v1.npz). Score 0 = looks like a camera photo, 1 = looks AI-generated.
The score is uncalibrated; fusion (pipeline/fusion.py) calibrates it together with the other signals.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .base import SignalResult
from .clip_features import clip_features

HEAD_PATH = Path(__file__).resolve().parents[1] / "models" / "s2_head_v1.npz"
_head = None


def _load_head() -> dict:
    global _head
    if _head is None:
        d = np.load(HEAD_PATH, allow_pickle=False)
        _head = {k: d[k] for k in ("coef", "intercept", "mean", "scale")}
        _head["meta"] = json.loads(str(d["meta"]))
    return _head


def score_features(x: np.ndarray) -> float:
    h = _load_head()
    logit = float(((x - h["mean"]) / h["scale"]) @ h["coef"] + h["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def run(evidence: Image.Image, reference=None, claim: str = "") -> SignalResult:
    """evidence must already be canonical (pipeline.canonical.canonicalise)."""
    t0 = time.time()
    s = score_features(clip_features(evidence))
    if s >= 0.8:
        reason = "Image statistics strongly resemble AI-generated photos"
    elif s >= 0.5:
        reason = "Image statistics partly resemble AI-generated photos"
    else:
        reason = "Image statistics look like a normal camera photo"
    return SignalResult(name="global_synthetic", score=round(s, 4), reason=reason,
                        artifacts={"head_version": _load_head()["meta"]["version"]},
                        latency_ms=int((time.time() - t0) * 1000))
