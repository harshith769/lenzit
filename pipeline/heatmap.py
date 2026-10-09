"""Heatmap overlay for the result page: PNG bytes of the canonical evidence image with a 0..1 mask on top."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def overlay_png(img: Image.Image, mask, alpha: float = 0.45) -> bytes | None:
    if mask is None:
        return None
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim != 2 or m.size == 0 or not np.isfinite(m).any():
        return None
    m = np.nan_to_num(m)
    w, h = img.size
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m, 0, 1)
    color = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    a = (alpha * m)[..., None]                       # only "hot" areas are tinted
    out = (base * (1 - a) + color.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, "PNG", optimize=True)
    return buf.getvalue()
