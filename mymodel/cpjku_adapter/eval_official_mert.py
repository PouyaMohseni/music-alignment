"""B1a eval: same official CPJKU eval pipeline as eval_official.py, but
loads precomputed MERT embeddings (from scripts/precompute_mert_zenodo.py)
instead of computing a live mel-spectrogram -- matching how B1a was trained
(extensions/hooks/mert_patch.py). Needed because B1a's checkpoint has
audio_encoder=MERTProjector, which expects (768, T) MERT features, not
CB_TA's native (78, T) mel-spec.

    python -m mymodel.cpjku_adapter.eval_official_mert \
        --cpjku_root  third_party/cpjku_unet \
        --cpjku_data  data/MSMD/cpjku_fmt \
        --processed   data/MSMD/processed \
        --param_path  <B1a best_model.pt> \
        --net_config  <B1a net_config.json> \
        --mert_root   /scratch/pmohseni/mert_emb_zenodo/msmd_test \
        --split       test
"""
from __future__ import annotations
import sys
from pathlib import Path

_MERT_ROOT = None


def _patched_load_piece_mert(params):
    """Same as eval_official._patched_load_piece, but substitutes a
    precomputed (768, T) MERT embedding for the live mel-spectrogram."""
    import os
    import numpy as np
    from scipy import interpolate
    from mymodel.cpjku_adapter import eval_official as _eo

    i, path, piece_name = params['i'], params['path'], params['piece_name']
    scale_factor = params.get('scale_factor', 1)
    spec_params  = params['spectrogram_params']

    npz = np.load(os.path.join(path, 'score', piece_name + '.npz'), allow_pickle=True)
    sheet        = npz['sheet']
    coords       = npz['coords'].astype(np.float32)
    onset_frames = npz['onset_frames']

    score = 1 - sheet.astype(np.float32) / 255.
    if scale_factor != 1:
        new_h = sheet.shape[0] // scale_factor
        new_w = sheet.shape[1] // scale_factor
        from PIL import Image as _Image
        resized = np.array(_Image.fromarray(sheet).resize((new_w, new_h), _Image.LANCZOS))
        score = 1 - resized.astype(np.float32) / 255.
        coords = coords / scale_factor

    emb_path = Path(_MERT_ROOT) / f'{piece_name}.npy'
    emb = np.load(emb_path).astype(np.float32)   # (T, 768)
    spec = emb.T                                  # (768, T) -- CPJKU spec convention
    spec = np.pad(spec, ((0, 0), (spec_params['pad'], 0)), mode='constant')

    onsets     = onset_frames
    coords_new = coords

    H_strip = sheet.shape[0] // scale_factor if scale_factor != 1 else sheet.shape[0]
    height_col = np.full((len(coords_new), 1), H_strip // 2, dtype=np.float32)
    coords_3 = np.concatenate([coords_new, height_col], axis=1)

    interpol_fnc = interpolate.interp1d(
        onsets, coords_3.T, kind='previous', bounds_error=False,
        fill_value=(coords_3[0, :], coords_3[-1, :]))

    unrolled_x = coords_new[:, 1]
    interpol_c2o = interpolate.interp1d(
        unrolled_x, onsets, kind='previous', bounds_error=False,
        fill_value=(onsets[0], onsets[-1]))

    staff_coords  = sorted(np.unique(coords_new[:, 0]))
    add_per_staff = np.array([0] * len(staff_coords))

    perf = {
        1000: {
            'interpol_fnc':  interpol_fnc,
            'spec':          spec,
            'onsets':        onsets,
            'interpol_c2o':  interpol_c2o,
            'add_per_staff': [staff_coords, add_per_staff],
        }
    }
    return i, score, piece_name, perf


def main():
    import argparse
    global _MERT_ROOT

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--mert_root', required=True)
    pre.add_argument('--cpjku_root', default='third_party/cpjku_unet')
    known, remaining = pre.parse_known_args()
    _MERT_ROOT = known.mert_root

    # audio_conditioned_unet isn't pip-installed in this venv (.venv, not
    # venv_cpjku310) -- eval_official.main() normally adds cpjku_root to
    # sys.path itself before importing it, but we need to import it here
    # first (to register MERTProjector) so that must happen before main() runs.
    cpjku_root = str(Path(known.cpjku_root).resolve())
    if cpjku_root not in sys.path:
        sys.path.insert(0, cpjku_root)

    from mymodel.cpjku_adapter import madmom_compat
    madmom_compat.patch()

    from mymodel.cpjku_adapter import eval_official as _eo
    _eo._patched_load_piece = _patched_load_piece_mert

    from audio_conditioned_unet import audio_encoder as _cpjku_audio_encoder
    from extensions.audio_encoders.mert_projector import MERTProjector
    _cpjku_audio_encoder.MERTProjector = MERTProjector

    sys.argv = [sys.argv[0]] + remaining
    _eo.main()


if __name__ == '__main__':
    main()
