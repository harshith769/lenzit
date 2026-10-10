"""S4 reference consistency (Harshith).

same_item_score(a, b): DINOv2 global similarity (1 = same item/product, 0 = different).
run(evidence, reference, claim) -> SignalResult "reference":
  1. same-item check (DINOv2 global embedding cosine)
  2. alignment: SIFT keypoints + ratio test + RANSAC homography (reference -> evidence frame)
  3. change map: aligned     -> 1 - SSIM on brightness-normalised grayscale
                 not aligned -> DINOv2 patch dissimilarity (coarse 16x16 fallback, any viewpoint)
  artifacts["change_mask"] (float32 HxW in [0,1], evidence size) feeds the S3 x S4 overlap feature.
S4 alone cannot tell fake damage from real damage (both differ from the reference); its score mainly
flags a different or reused item (T0). Thresholds are uncalibrated until Lenzit-Bench pairs exist.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms as T

from .base import SignalResult

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"   # Apache-2.0
SAME_ITEM_T = 0.2   # calibrated on Lenzit-Bench v1 T0 pairs (models/decision_v1.json)
MIN_INLIERS = 25
_TF = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),   # whole frame, item never cropped out
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
_model = None
_sift = None


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
    """1.0 = same physical item/product, 0.0 = different (cosine of DINOv2 embeddings, clipped)."""
    return float(np.clip(np.dot(embed(a), embed(b)), 0.0, 1.0))


@torch.no_grad()
def _patch_tokens(img: Image.Image) -> np.ndarray:
    m = _load()
    t = m.forward_features(_TF(img.convert("RGB")).unsqueeze(0).to(DEVICE))[:, m.num_prefix_tokens:]
    return torch.nn.functional.normalize(t, dim=-1)[0].cpu().numpy()   # (256, 768) at 224 px


def _gray(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)


def align(reference: Image.Image, evidence: Image.Image):
    """Homography mapping reference pixels -> evidence pixels, or None. Returns (H, inlier_count)."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=4000)
    k1, d1 = _sift.detectAndCompute(_gray(reference), None)
    k2, d2 = _sift.detectAndCompute(_gray(evidence), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    pairs = [p for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2) if len(p) == 2]
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < MIN_INLIERS:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < MIN_INLIERS or not 0.25 < abs(np.linalg.det(H[:2, :2])) < 4.0:
        return None, inliers
    return H, inliers


def _ssim_change(ev_gray: np.ndarray, ref_warp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    a, b = ev_gray.astype(np.float32), ref_warp.astype(np.float32)
    for x in (a, b):  # brightness/contrast normalisation inside the overlap (photos differ in light)
        x -= x[valid].mean()
        x /= x[valid].std() + 1e-6
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    s_aa, s_bb, s_ab = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2, blur(a * b) - mu_a * mu_b
    c1, c2 = 0.06 ** 2, 0.18 ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    change = cv2.GaussianBlur(np.clip((1 - ssim) / 2, 0, 1), (0, 0), 5)
    change[~valid] = 0
    return change


def _patch_change(evidence: Image.Image, reference: Image.Image, size) -> np.ndarray:
    e, r = _patch_tokens(evidence), _patch_tokens(reference)
    best = (e @ r.T).max(axis=1)                  # best match anywhere in the reference, per evidence patch
    n = int(round(len(best) ** 0.5))
    change = np.clip(1 - best, 0, 1).reshape(n, n).astype(np.float32)
    return np.clip(cv2.resize(change, size, interpolation=cv2.INTER_CUBIC), 0, 1)


def run(evidence: Image.Image, reference: Image.Image | None = None, claim: str = "") -> SignalResult:
    """evidence and reference must already be canonical (pipeline.canonical.canonicalise)."""
    t0 = time.time()
    w, h = evidence.size
    if reference is None:
        return SignalResult("reference", 0.5, "No reference photo was provided, so the item could not be compared",
                            {"same_item": None, "aligned": False, "inliers": 0, "method": "none",
                             "change_mask": np.zeros((h, w), np.float32)}, int((time.time() - t0) * 1000))
    same = same_item_score(evidence, reference)
    H, inliers = align(reference, evidence)
    if H is not None:
        ref_warp = cv2.warpPerspective(_gray(reference), H, (w, h))
        valid = cv2.warpPerspective(np.ones((reference.size[1], reference.size[0]), np.uint8), H, (w, h))
        valid = cv2.erode(valid, np.ones((15, 15), np.uint8)).astype(bool)
        change, method = _ssim_change(_gray(evidence), ref_warp, valid), "homography+ssim"
    else:
        change, method = _patch_change(evidence, reference, (w, h)), "dinov2-patches"
    if same < SAME_ITEM_T:
        score, reason = 0.9, "The photo does not appear to show the same item as the seller's reference"
    else:
        score, reason = round(1 - same, 4), "Same item as the seller's reference; areas that differ are highlighted"
    return SignalResult("reference", score, reason,
                        {"same_item": round(same, 4), "aligned": H is not None, "inliers": inliers,
                         "method": method, "change_mask": change.astype(np.float32)},
                        int((time.time() - t0) * 1000))
