"""Shared contract every Lenzit signal implements. Agreed Oct 4; change only via the source-of-truth doc."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol
from PIL import Image


@dataclass
class SignalResult:
    name: str                 # e.g. "global_synthetic"
    score: float              # 0 = looks genuine, 1 = looks manipulated
    reason: str               # one plain-language sentence
    artifacts: dict = field(default_factory=dict)  # e.g. {"heatmap_png": bytes}
    latency_ms: int = 0


class Signal(Protocol):
    name: str
    def run(self, evidence: Image.Image, reference: Optional[Image.Image], claim: str) -> SignalResult: ...
