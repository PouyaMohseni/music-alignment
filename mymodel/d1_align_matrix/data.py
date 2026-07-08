"""D1 dataset. Per piece, load:
  - MERT embedding (T, 768) fp16 -> fp32, whole-piece, keyed by piece name.
  - strip score + alignment geometry directly from cpjku_fmt/score/<piece>.npz
    (sheet, coords=[y,x,h], onset_frames). We replicate ONLY the geometry that
    eval_official._patched_load_piece builds (score resize, interpol_fnc,
    interpol_c2o, add_per_staff) and deliberately SKIP its spectrogram step --
    D1 substitutes the precomputed MERT embedding for audio, so touching the wav
    would be wasted work and an extra dependency.
  - dense GT column per frame: interpol_fnc(frame)[1] (strip-x) / w_downsample.

The strip (unrolled score of the piece) and the MERT embedding (the piece's
performance audio) describe the SAME whole piece; coords/onsets tie strip-x to
performance frame. Only tempo_1000 whole-piece is used for the first D1 run
(see D1.md sec 6).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from scipy import interpolate


def _resize_strip(sheet: np.ndarray, scale_factor: int) -> np.ndarray:
    """1 - sheet/255, downscaled by scale_factor. PIL (always available) instead
    of cv2 to avoid the cv2-stub issue on machines without the opencv module."""
    from PIL import Image
    H, W = sheet.shape
    new_w, new_h = W // scale_factor, H // scale_factor
    resized = np.array(Image.fromarray(sheet).resize((new_w, new_h), Image.LANCZOS))
    return 1.0 - resized.astype(np.float32) / 255.0


def _mert_path(mert_roots, piece: str) -> str | None:
    for root in mert_roots:
        p = Path(root) / f'{piece}.npy'
        if p.exists():
            return str(p)
    return None


class D1Piece:
    def __init__(self, piece_name, mert, strip, gt_cols, onset_frames,
                 interpol_c2o, add_per_staff, w_downsample):
        self.piece_name = piece_name
        self.mert = mert                      # (T, 768) float32 tensor
        self.strip = strip                    # (1, 1, H, W) float32 tensor
        self.gt_cols = gt_cols                # (T,) long tensor
        self.onset_frames = onset_frames      # (n_onsets,) int64, frames with GT
        self.interpol_c2o = interpol_c2o      # callable strip-x -> onset time (frames)
        self.add_per_staff = add_per_staff    # scalar x offset (single-staff strips)
        self.w_downsample = w_downsample


def load_piece(piece_name, cpjku_data, mert_roots, scale_factor, w_downsample):
    """Returns a D1Piece, or None if MERT embedding or score npz is missing."""
    mp = _mert_path(mert_roots, piece_name)
    if mp is None:
        return None
    npz_path = Path(cpjku_data) / 'score' / f'{piece_name}.npz'
    if not npz_path.exists():
        return None

    mert = np.load(mp).astype(np.float32)      # (T, 768)
    T = mert.shape[0]

    z = np.load(npz_path, allow_pickle=True)
    sheet = z['sheet']
    coords = z['coords'].astype(np.float32).copy()
    onset_frames_raw = z['onset_frames']

    score = _resize_strip(sheet, scale_factor)     # (H, W) in [0,1]
    coords /= scale_factor
    H, W = score.shape
    W_col = int(np.ceil(W / w_downsample))

    # interpol_fnc: onset frame -> [y, x, height]; 'previous' hold + clamp fill,
    # exactly as eval_official._patched_load_piece builds it.
    H_strip = H
    height_col = np.full((len(coords), 1), H_strip // 2, dtype=np.float32)
    coords_3 = np.concatenate([coords, height_col], axis=1)   # (N,3)
    onsets = onset_frames_raw
    interpol_fnc = interpolate.interp1d(onsets, coords_3.T, kind='previous',
                                        bounds_error=False,
                                        fill_value=(coords_3[0], coords_3[-1]))
    unrolled_x = coords[:, 1]
    interpol_c2o = interpolate.interp1d(unrolled_x, onsets, kind='previous',
                                        bounds_error=False,
                                        fill_value=(onsets[0], onsets[-1]))
    staff_coords = sorted(np.unique(coords[:, 0]))
    add_per_staff = np.array([0] * len(staff_coords))

    gt_x = np.array([np.asarray(interpol_fnc(t))[1] for t in range(T)], dtype=np.float32)
    gt_cols = np.clip(np.round(gt_x / w_downsample).astype(np.int64), 0, W_col - 1)

    onset_frames = np.array(sorted(int(f) for f in set(int(x) for x in onsets) if 0 <= int(f) < T),
                            dtype=np.int64)

    return D1Piece(
        piece_name=piece_name,
        mert=torch.from_numpy(mert),
        strip=torch.from_numpy(score[np.newaxis, np.newaxis]),
        gt_cols=torch.from_numpy(gt_cols),
        onset_frames=onset_frames,
        interpol_c2o=interpol_c2o,
        add_per_staff=float(add_per_staff[0]) if len(add_per_staff) else 0.0,
        w_downsample=w_downsample,
    )


def load_split(split, processed_root, cpjku_data, mert_roots, scale_factor,
               w_downsample, limit=None):
    import json
    splits = json.load(open(Path(processed_root) / 'splits.json'))
    names = splits.get(split, [])
    if limit is not None:
        names = names[:limit]
    pieces, skipped = [], []
    for name in names:
        p = load_piece(name, cpjku_data, mert_roots, scale_factor, w_downsample)
        (pieces if p is not None else skipped).append(p if p is not None else name)
    print(f'[D1Dataset] split={split}: loaded {len(pieces)}, skipped {len(skipped)} '
          f'(missing score or MERT)', flush=True)
    return pieces
