"""S4 reference consistency (Harshith).

Step 1 (now): same_item_score via DINOv2 global embeddings.
Step 2 (Oct 6): alignment (LightGlue/DISK + RANSAC) and change mask -> run() returning SignalResult.
"""
from __future__ import annotations

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"   # Apache-2.0
_TF = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),   # whole frame, so the item is never cropped out
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
_model = None


def _load():
    global _model
    if _model is None:
        _model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, img_size=224).eval().to(DEVICE)
    return _model


@torch.no_grad()
def embed(img: Image.Image) -> np.ndarray:
    x = _TF(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    f = _load()(x)
    return torch.nn.functional.normalize(f, dim=-1)[0].cpu().numpy()


def same_item_score(a: Image.Image, b: Image.Image) -> float:
    """1.0 = same physical item/product, 0.0 = different.
    Cosine similarity of DINOv2 embeddings, clipped to [0, 1]. Uncalibrated: the
    'same item' threshold is fit on Lenzit-Bench pairs once labels exist."""
    return float(np.clip(np.dot(embed(a), embed(b)), 0.0, 1.0))
