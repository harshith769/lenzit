"""One deterministic transform used identically in training and inference (source-of-truth doc, section 8).

Every image (real or fake) becomes a 1024x768 (or 768x1024) JPEG-q90 RGB image, so file size,
aspect ratio and encoder fingerprints cannot leak the label. S1 reads metadata BEFORE this runs.
"""
from io import BytesIO
from PIL import Image, ImageOps

LONG_SIDE, SHORT_SIDE = 1024, 768   # 4:3
JPEG_QUALITY = 90


def canonicalise(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")      # apply phone rotation, drop alpha
    w, h = img.size
    landscape = w >= h
    target = LONG_SIDE / SHORT_SIDE if landscape else SHORT_SIDE / LONG_SIDE
    if w / h > target:                                       # too wide -> crop sides
        nw = round(h * target); left = (w - nw) // 2
        img = img.crop((left, 0, left + nw, h))
    else:                                                    # too tall -> crop top/bottom
        nh = round(w / target); top = (h - nh) // 2
        img = img.crop((0, top, w, top + nh))
    size = (LONG_SIDE, SHORT_SIDE) if landscape else (SHORT_SIDE, LONG_SIDE)
    img = img.resize(size, Image.Resampling.LANCZOS)         # exact size, up or down
    buf = BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)              # same encoder for every image
    buf.seek(0)
    return Image.open(buf).convert("RGB")
