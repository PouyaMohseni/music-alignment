"""Minimal cv2 stub for Alliance Canada cluster (Python 3.10 venv).

The system OpenCV is compiled for Python 3.11 and is ABI-incompatible with
this 3.10 venv.  We provide the subset of cv2 actually used by the CPJKU
audio_conditioned_unet code (utils.py, video_utils.py) via PIL and numpy.
"""
import numpy as np
from PIL import Image as _Image

# ── Interpolation flags ───────────────────────────────────────────────────────
INTER_NEAREST = 0
INTER_LINEAR  = 1
INTER_AREA    = 2
INTER_CUBIC   = 3

# ── Color conversion codes ────────────────────────────────────────────────────
COLOR_GRAY2BGR  = 8
COLOR_GRAY2RGB  = 8
COLOR_BGR2GRAY  = 6
COLOR_RGB2GRAY  = 7
COLOR_RGB2BGRA  = 26
COLOR_RGBA2BGRA = 26
COLOR_BGR2RGB   = 4


def resize(src, dsize, dst=None, fx=None, fy=None, interpolation=INTER_LINEAR):
    """Resize image using PIL (supports all interpolation flags)."""
    h, w = dsize[1], dsize[0]
    pil_map = {
        INTER_NEAREST: _Image.NEAREST,
        INTER_LINEAR:  _Image.BILINEAR,
        INTER_AREA:    _Image.LANCZOS,
        INTER_CUBIC:   _Image.BICUBIC,
    }
    resample = pil_map.get(interpolation, _Image.BILINEAR)

    if src.ndim == 2:
        if src.dtype == np.float32 or src.dtype == np.float64:
            arr8 = (src * 255).clip(0, 255).astype(np.uint8)
            out8 = np.array(_Image.fromarray(arr8, mode='L').resize((w, h), resample))
            return (out8.astype(src.dtype) / 255.0)
        return np.array(_Image.fromarray(src.astype(np.uint8), mode='L').resize((w, h), resample))
    elif src.ndim == 3:
        mode = 'RGB' if src.shape[2] == 3 else 'RGBA'
        return np.array(_Image.fromarray(src.astype(np.uint8), mode=mode).resize((w, h), resample))
    raise ValueError(f"Unsupported src.ndim={src.ndim}")


def cvtColor(src, code):
    """Color space conversion (subset used by CPJKU code)."""
    if code in (COLOR_GRAY2BGR, COLOR_GRAY2RGB):
        if src.ndim == 2:
            return np.stack([src, src, src], axis=-1)
        return src
    if code == COLOR_BGR2GRAY:
        return (0.114 * src[..., 0] + 0.587 * src[..., 1] + 0.299 * src[..., 2]).astype(src.dtype)
    if code == COLOR_RGB2GRAY:
        return (0.299 * src[..., 0] + 0.587 * src[..., 1] + 0.114 * src[..., 2]).astype(src.dtype)
    if code in (COLOR_RGB2BGRA, COLOR_RGBA2BGRA):
        if src.shape[-1] == 3:
            alpha = np.full(src.shape[:2] + (1,), 255, dtype=src.dtype)
            bgr = src[..., ::-1]
            return np.concatenate([bgr, alpha], axis=-1)
        # RGBA → BGRA: flip R↔B
        out = src.copy()
        out[..., 0], out[..., 2] = src[..., 2].copy(), src[..., 0].copy()
        return out
    if code == COLOR_BGR2RGB:
        return src[..., ::-1].copy()
    raise NotImplementedError(f"cvtColor code {code} not implemented in stub")


def addWeighted(src1, alpha, src2, beta, gamma):
    return np.clip(src1 * alpha + src2 * beta + gamma, 0, 255).astype(src1.dtype)


def imshow(winname, mat):
    pass  # headless — no display


def waitKey(delay=0):
    return -1


class VideoWriter:
    def __init__(self, *a, **kw): pass
    def write(self, frame): pass
    def release(self): pass
    @staticmethod
    def isOpened(): return False


def VideoWriter_fourcc(*args):
    return 0
