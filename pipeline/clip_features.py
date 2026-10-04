"""S2 backbone: frozen CLIP ViT-L/14 features from native-resolution crops of the canonical image.

Native 224px crops (4 corners + centre) keep the pixel-level traces of AI generation that a
whole-image resize would smooth away. The 5 crop embeddings are L2-normalised and averaged.
"""
from __future__ import annotations
import numpy as np
import open_clip
import torch
from PIL import Image

CROP = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)  # CLIP normalisation
_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
_model = None


def _load():
    global _model
    if _model is None:
        m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
        _model = m.eval().to(DEVICE)
    return _model


def five_crops(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    corners = [(0, 0), (w - CROP, 0), (0, h - CROP), (w - CROP, h - CROP), ((w - CROP) // 2, (h - CROP) // 2)]
    return [img.crop((x, y, x + CROP, y + CROP)) for x, y in corners]


def _to_tensor(crop: Image.Image) -> torch.Tensor:
    a = torch.from_numpy(np.array(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (a - _MEAN) / _STD


@torch.no_grad()
def clip_features(img: Image.Image) -> np.ndarray:
    """img must already be canonical (pipeline.canonical.canonicalise). Returns a 768-d vector."""
    model = _load()
    batch = torch.stack([_to_tensor(c) for c in five_crops(img)]).to(DEVICE)
    f = model.encode_image(batch)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.mean(0).float().cpu().numpy()
