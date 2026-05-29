"""Stage 3 + Stage 4 of the MSMD preprocessing pipeline.

Given one score engraving directory, this module produces:
  - the unrolled strip image (PIL.Image)
  - the strip-to-page mapping table
"""
from __future__ import annotations
import glob, os
from dataclasses import dataclass
import numpy as np
from PIL import Image


DEFAULT_MAX_PAD_PX     = 40
DEFAULT_TARGET_RAW_H   = 107
DEFAULT_OUTPUT_H       = 224   # ViT-native tile height


@dataclass
class _System:
    page_idx: int
    sys_idx_on_page: int
    raw_bbox: tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max) on page
    exp_bbox: tuple[int, int, int, int]


def _expand_one_page(systems_npy_path: str, page_h: int, max_pad_px: int):
    arr = np.load(systems_npy_path)
    arr = arr[np.argsort(arr[:, 0, 0])]
    raw = [(int(c[:, 1].min()), int(c[:, 0].min()),
            int(c[:, 1].max()), int(c[:, 0].max())) for c in arr]
    out = []
    for i, (xmn, ymn, xmx, ymx) in enumerate(raw):
        mid_top = 0      if i == 0           else (raw[i - 1][3] + ymn) // 2
        mid_bot = page_h if i == len(raw) - 1 else (ymx + raw[i + 1][1]) // 2
        top = max(mid_top, ymn - max_pad_px, 0)
        bot = min(mid_bot, ymx + max_pad_px, page_h)
        out.append(((xmn, ymn, xmx, ymx), (xmn, top, xmx, bot)))
    return out


def collect_systems(score_dir: str, max_pad_px: int = DEFAULT_MAX_PAD_PX) -> list[_System]:
    """Walk all pages of a score engraving and return systems in reading order."""
    systems = []
    for page_idx, sys_path in enumerate(sorted(
            glob.glob(os.path.join(score_dir, "coords", "systems_*.npy")))):
        img_path = os.path.join(score_dir, "img", f"{page_idx + 1:02d}.png")
        with Image.open(img_path) as img:
            page_h = img.size[1]
        for sys_idx, (raw, exp) in enumerate(
                _expand_one_page(sys_path, page_h, max_pad_px)):
            systems.append(_System(page_idx, sys_idx, raw, exp))
    return systems


def build_strip(score_dir: str,
                max_pad_px: int = DEFAULT_MAX_PAD_PX,
                target_raw_h: int = DEFAULT_TARGET_RAW_H,
                output_height: int = DEFAULT_OUTPUT_H):
    """Build the unrolled strip image for one score engraving.

    The strip is padded vertically with white so its final height equals
    output_height (224 by default, ViT-native). The staff is centred so that
    horizontally tiled ViT crops require no further padding.

    Returns
    -------
    strip : PIL.Image
    mapping : list[dict]
        One entry per system, in strip order, with keys
        strip_x_start, strip_x_end, page_idx, system_idx_on_page,
        raw_bbox, exp_bbox, scale.
    """
    systems = collect_systems(score_dir, max_pad_px)

    # First pass: crop, scale to target raw height, record top/bottom margins.
    crops = []
    metas = []
    page_cache = {}
    for s in systems:
        if s.page_idx not in page_cache:
            page_cache[s.page_idx] = Image.open(
                os.path.join(score_dir, "img", f"{s.page_idx + 1:02d}.png")
            ).convert("RGB")
        img = page_cache[s.page_idx]
        xmn, ymn_e, xmx, ymx_e = s.exp_bbox
        _, ymn_r, _, ymx_r     = s.raw_bbox
        crop = img.crop((xmn, ymn_e, xmx, ymx_e))
        scale = target_raw_h / (ymx_r - ymn_r)
        crop = crop.resize(
            (crop.size[0], int(round(crop.size[1] * scale))),
            Image.LANCZOS,
        )
        top_m = int(round((ymn_r - ymn_e) * scale))
        bot_m = int(round((ymx_e - ymx_r) * scale))
        crops.append(crop)
        metas.append((s, scale, top_m, bot_m))

    max_top = max(m[2] for m in metas)
    max_bot = max(m[3] for m in metas)
    inner_h = max_top + target_raw_h + max_bot
    if inner_h > output_height:
        raise ValueError(
            f"expanded strip height {inner_h} exceeds output_height {output_height}; "
            "reduce target_raw_h or max_pad_px"
        )
    extra_top = (output_height - inner_h) // 2
    total_w = sum(c.size[0] for c in crops)

    strip = Image.new("RGB", (total_w, output_height), "white")
    mapping = []
    x = 0
    for crop, (s, scale, top_m, _) in zip(crops, metas):
        strip.paste(crop, (x, extra_top + max_top - top_m))
        mapping.append({
            "strip_x_start":       x,
            "strip_x_end":         x + crop.size[0],
            "page_idx":            s.page_idx,
            "system_idx_on_page":  s.sys_idx_on_page,
            "raw_bbox":            list(s.raw_bbox),
            "exp_bbox":            list(s.exp_bbox),
            "scale":               float(scale),
        })
        x += crop.size[0]
    return strip, mapping
