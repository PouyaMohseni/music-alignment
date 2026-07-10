"""E4 -- tempo-variant D2 piece loader. Extends D2Piece to a GIVEN tempo_factor
(default 1000, delegating to the original single-tempo loader unchanged for
that case) using whole-piece MIDI/audio rendered by
scripts/render_multitempo_wholepiece.py and MERT-encoded from those renders.

Score/coords geometry is tempo-INDEPENDENT (same score image regardless of
playback speed) -- only the FRAME axis changes, since fps is fixed at 20 but
the audio now plays faster/slower. Tempo-scaling convention (verified in
extensions/pretrain/tempo_contrastive.py against real per-page multi-tempo
MSMD renders, and independently re-verified here on real whole-piece renders
-- see this module's __main__ smoke test): onset time (and therefore onset
FRAME, at fixed fps) scales by tempo_factor/1000 relative to the tempo_1000
base (tempo_750 = all times at 0.75x = notes closer together = faster).

Repeat detection (mymodel.d2_midi_privileged.repeat_labels) is tempo-
INDEPENDENT by construction -- it only uses relative pitch order/intervals,
matched positionally to coords/onset_frames -- so it is computed ONCE from
the tempo_1000 MIDI/onsets exactly as D2's original loader does, and the
resulting {column: [alternates]} mapping is reused unchanged for every tempo
variant (only the FRAME position at which each onset's alternates apply
changes, via the rescaled onset frames).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from scipy import interpolate

from mymodel.d1_align_matrix.data import D1Piece, _resize_strip
from mymodel.d2_midi_privileged.data import D2Piece, load_piece as d2_load_piece_base
from mymodel.d2_midi_privileged.midi_encoder import compute_pitch_roll
from mymodel.d2_midi_privileged.repeat_labels import build_repeat_groups_for_piece


def _mert_path_tempo(mert_roots_multitempo, piece: str, tempo_factor: int) -> str | None:
    for root in mert_roots_multitempo:
        p = Path(root) / f'{piece}_tempo_{tempo_factor}.npy'
        if p.exists():
            return str(p)
    return None


def load_piece_multitempo(piece_name, cpjku_data, mert_roots, mert_roots_multitempo,
                          multitempo_render_dir, scale_factor, w_downsample,
                          tempo_factor=1000, repeat_k=5):
    """tempo_factor=1000 delegates entirely to D2's original (unchanged) loader.
    Otherwise: same score/repeat-group logic, rescaled onset-frame axis, and
    pitch_roll built from the TEMPO-SCALED rendered MIDI (multitempo_render_dir/
    <piece>_tempo_<T>.mid) so its note timing matches this tempo's actual frame
    count -- NOT the base tempo_1000 MIDI, which would misalign pitch_roll
    against the (shorter/longer) MERT sequence at this tempo."""
    if tempo_factor == 1000:
        return d2_load_piece_base(piece_name, cpjku_data, mert_roots, scale_factor,
                                  w_downsample, repeat_k)

    mp = _mert_path_tempo(mert_roots_multitempo, piece_name, tempo_factor)
    if mp is None:
        return None
    scaled_midi_path = Path(multitempo_render_dir) / f'{piece_name}_tempo_{tempo_factor}.mid'
    base_midi_path = Path(cpjku_data) / 'performance' / f'{piece_name}.mid'
    if not scaled_midi_path.exists() or not base_midi_path.exists():
        return None
    npz_path = Path(cpjku_data) / 'score' / f'{piece_name}.npz'
    if not npz_path.exists():
        return None

    mert = np.load(mp).astype(np.float32)   # (T_new, 768)
    T = mert.shape[0]

    z = np.load(npz_path, allow_pickle=True)
    sheet = z['sheet']
    coords = z['coords'].astype(np.float32).copy()
    onset_frames_base = z['onset_frames']            # tempo_1000 frame units

    score = _resize_strip(sheet, scale_factor)
    coords /= scale_factor
    H, W = score.shape
    W_col = int(np.ceil(W / w_downsample))

    ratio = tempo_factor / 1000.0
    onset_frames_scaled = onset_frames_base.astype(np.float64) * ratio   # this tempo's frame units

    height_col = np.full((len(coords), 1), H // 2, dtype=np.float32)
    coords_3 = np.concatenate([coords, height_col], axis=1)
    interpol_fnc = interpolate.interp1d(onset_frames_scaled, coords_3.T, kind='previous',
                                        bounds_error=False,
                                        fill_value=(coords_3[0], coords_3[-1]))
    unrolled_x = coords[:, 1]
    interpol_c2o = interpolate.interp1d(unrolled_x, onset_frames_scaled, kind='previous',
                                        bounds_error=False,
                                        fill_value=(onset_frames_scaled[0], onset_frames_scaled[-1]))
    staff_coords = sorted(np.unique(coords[:, 0]))
    add_per_staff = np.array([0] * len(staff_coords))

    gt_x = np.array([np.asarray(interpol_fnc(t))[1] for t in range(T)], dtype=np.float32)
    gt_cols = np.clip(np.round(gt_x / w_downsample).astype(np.int64), 0, W_col - 1)

    onset_frames_t = np.array(
        sorted(int(f) for f in set(int(round(x)) for x in onset_frames_scaled) if 0 <= f < T),
        dtype=np.int64)

    d1piece = D1Piece(
        piece_name=piece_name,
        mert=torch.from_numpy(mert),
        strip=torch.from_numpy(score[np.newaxis, np.newaxis]),
        gt_cols=torch.from_numpy(gt_cols),
        onset_frames=onset_frames_t,
        interpol_c2o=interpol_c2o,
        add_per_staff=float(add_per_staff[0]) if len(add_per_staff) else 0.0,
        w_downsample=w_downsample,
    )

    # Repeat groups: tempo-independent, computed from the BASE tempo_1000 MIDI
    # and BASE onset_frames (matches how repeat_labels.py was designed/tested).
    col_alternates = build_repeat_groups_for_piece(
        str(base_midi_path), coords, onset_frames_base, w_downsample, k=repeat_k)
    repeat_alt_cols = [[] for _ in range(T)]
    onset_to_true_col = {}
    for i, f_scaled in enumerate(onset_frames_scaled):
        f = int(round(f_scaled))
        if 0 <= f < T:
            onset_to_true_col[f] = int(round(coords[i, 1] / w_downsample))
    for f, true_col in onset_to_true_col.items():
        if true_col in col_alternates:
            repeat_alt_cols[f] = col_alternates[true_col]

    # pitch_roll from the TEMPO-SCALED rendered MIDI (correct timing at this tempo).
    pitch_roll = compute_pitch_roll(str(scaled_midi_path), T, fps=20)

    return D2Piece(d1piece, torch.from_numpy(pitch_roll), repeat_alt_cols)


def load_split_multitempo(split, processed_root, cpjku_data, mert_roots, mert_roots_multitempo,
                          multitempo_render_dir, scale_factor, w_downsample,
                          tempo_factors=(1000,), repeat_k=5, limit=None):
    """Loads (piece, tempo) pairs across all given tempo_factors -- each is an
    independent training example (same piece, different playback speed)."""
    import json
    splits = json.load(open(Path(processed_root) / 'splits.json'))
    names = splits.get(split, [])
    if limit is not None:
        names = names[:limit]
    pieces, skipped = [], []
    for name in names:
        for tf in tempo_factors:
            p = load_piece_multitempo(name, cpjku_data, mert_roots, mert_roots_multitempo,
                                      multitempo_render_dir, scale_factor, w_downsample,
                                      tempo_factor=tf, repeat_k=repeat_k)
            (pieces if p is not None else skipped).append(p if p is not None else f'{name}_tempo_{tf}')
    print(f'[D2MultitempoDataset] split={split} tempo_factors={tempo_factors}: '
          f'loaded {len(pieces)}, skipped {len(skipped)}', flush=True)
    return pieces
