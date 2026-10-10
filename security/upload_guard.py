"""Upload guard: reject anything that is not a sane, decodable image before it reaches the pipeline.

validate_upload(data, max_bytes=12_000_000, max_pixels=50_000_000) -> PIL.Image.Image
Raises UploadRejected; .status_code is the HTTP status serving/api.py should return:
  413 = too large (bytes or pixels / decompression bomb), 422 = not a supported/decodable image.
"""
from __future__ import annotations

import io
import warnings

from PIL import Image

try:  # HEIC/HEIF support (iPhone photos)
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HEIF_OK = True
except Exception:  # pragma: no cover - only if pillow-heif missing
    _HEIF_OK = False

DEFAULT_MAX_BYTES = 12_000_000
DEFAULT_MAX_PIXELS = 50_000_000


class UploadRejected(Exception):
    def __init__(self, reason: str, status_code: int = 422):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


def _sniff(data: bytes) -> str | None:
    """Identify the format from magic bytes only. File names and client Content-Type are ignored."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:8] == b"ftyp" and data[8:12] in {b"heic", b"heix", b"heim", b"heis", b"mif1", b"msf1", b"hevc"}:
        return "image/heic"
    return None


def validate_upload(
    data: bytes,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> Image.Image:
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        raise UploadRejected("Empty upload.", 422)
    if len(data) > max_bytes:
        raise UploadRejected(f"File is larger than the {max_bytes // 1_000_000} MB limit.", 413)

    mime = _sniff(bytes(data[:32]))
    if mime is None:
        raise UploadRejected("Unsupported file type. Upload a JPEG, PNG, WebP or HEIC photo.", 422)
    if mime == "image/heic" and not _HEIF_OK:
        raise UploadRejected("HEIC photos are not supported on this server.", 422)

    # Decompression-bomb guard: check declared dimensions from the header before any full decode,
    # and make Pillow's own bomb check use our limit while we decode.
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                w, h = probe.size
                if w <= 0 or h <= 0:
                    raise UploadRejected("Invalid image dimensions.", 422)
                if w * h > max_pixels:
                    raise UploadRejected(f"Image is larger than the {max_pixels // 1_000_000} MP limit.", 413)
                probe.verify()  # structural check; probe is unusable after this

            img = Image.open(io.BytesIO(data))
            img.load()  # full decode now that size is known to be safe
            return img
    except UploadRejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise UploadRejected("Image exceeds the pixel limit.", 413)
    except Exception:
        raise UploadRejected("File is not a valid image.", 422)
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
