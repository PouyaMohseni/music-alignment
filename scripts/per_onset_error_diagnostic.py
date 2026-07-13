"""Per-ONSET (not just per-piece) error diagnostic for our best model (F4:
v13+v14+v15 ensemble + hybrid_snap decode). For each of the worst/shared
pieces (per the cross-model failure analysis), computes error at every
individual onset and tests it against the two known failure-mechanism
hypotheses directly, using data we fully control (no cross-pipeline
alignment risk against CPJKU's own onset bookkeeping):

  1. Repeat-ambiguity: is this onset's score column within a known
     repeat-equivalent group (build_repeat_alt_cols, same tooling as
     E2/E3/F6)?
  2. Sparse audio: what's the local onset density (notes/sec) in a window
     around this onset?

Reports, per piece and in aggregate: correlation between per-onset error and
each flag/covariate, plus WHERE in the piece (relative position 0-1) the
worst onsets cluster.

    python scripts/per_onset_error_diagnostic.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mymodel.f3_ensemble_decode.eval import (
    MODEL_REGISTRY, _load_model, _predict_x_com_from_heatmap,
    _decode_offline_dtw, _build_perf_frame,
)
from mymodel.v11_cpjku_fullstrip.data import load_strip_scaled
from mymodel.v13_midi_privileged.repeat_gt import build_repeat_alt_cols

WORST_SHARED_PIECES = [
    'ChopinFF__O28__Chop-28-9', 'MussorgskyM__pictures-at-an-exhibition__catacombae',
    'SchumannR__O68__schumann-op68-01-melodie', 'ScriabinA__O16__scriabine-op16-no5',
    'SatieE__gymnopedie_3__gymnopedie_3', 'SatieE__gymnopedie_1__gymnopedie_1',
    'SchumannR__O68__schumann-op68-04-choral', 'CzernyC__Op_821__Czerny_Op_821_No_014',
    'MussorgskyM__pictures-at-an-exhibition__promenade-3', 'ChopinFF__O9__nocturne_in_b-flat_minor',
    'SchumannR__O68__schumann-op68-06-pauvre-orpheline', 'SchumannR__O68__schumann-op68-09-chanson-populaire',
    'BachJS__BWV825__16title-hub', 'BartokB__rom_folk_dance_1_bartok__rom_folk_dance_1_bartok',
]


@torch.no_grad()
def run_piece(pid: str, loaded, processed_root: Path, emb_root: Path, device: str,
              w_scale: int, h_strip: int, fps: int, snap_frac: float = 0.2):
    piece_dir = processed_root / pid
    notes = np.load(piece_dir / 'noteheads.npz')
    feats = np.load(str(emb_root / f'{pid}.npy')).astype(np.float32)
    T = feats.shape[0]

    score = load_strip_scaled(piece_dir / 'strip.png', h_strip, w_scale)
    H, W_sc = score.shape
    score_1 = torch.from_numpy(score[np.newaxis, np.newaxis, np.newaxis]).to(device)

    hiddens = []
    for _, network, _ in loaded:
        if network.use_lstm:
            hiddens.append((torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device),
                            torch.zeros(network.rnn_layers, 1, network.rnn_size, device=device)))
        else:
            hiddens.append(None)

    pred_x_sc_orig = np.zeros(T, dtype=np.float64)
    marginals = np.zeros((T, W_sc), dtype=np.float64)
    for t in range(T):
        avg = np.zeros((H, W_sc), dtype=np.float32)
        for i, (name, network, cfg) in enumerate(loaded):
            n_frames = cfg.data.n_frames
            perf_np = _build_perf_frame(feats, t, n_frames)
            perf_t = torch.from_numpy(perf_np).to(device).unsqueeze(0)
            out = network(score=score_1, perf=perf_t, hidden=hiddens[i])
            seg = out['segmentation']
            new_hidden = out.get('hidden')
            if new_hidden is not None:
                hiddens[i] = (new_hidden[0].detach(), new_hidden[1].detach())
            avg += (1.0 / len(loaded)) * seg.squeeze(0).squeeze(0).cpu().numpy()
        pred_x_sc_orig[t] = _predict_x_com_from_heatmap(avg)
        marginals[t] = avg.sum(axis=0)

    dtw_path = _decode_offline_dtw(marginals, 0.05)
    diff = np.abs(pred_x_sc_orig - dtw_path)
    hybrid = np.where(diff > snap_frac * W_sc, dtw_path, pred_x_sc_orig)
    pred_x_orig = hybrid * w_scale

    onset_sec = notes['onset_sec']
    gt_strip_x = notes['strip_x']
    midi_pitch = notes['midi_pitch']
    frame_idx = np.clip(np.round(onset_sec * fps).astype(int), 0, T - 1)
    pred_at_onset = pred_x_orig[frame_idx]

    ann = json.load(open(piece_dir / 'annotations.json'))
    dur = float(ann['audio']['duration_sec'])
    strip_w = ann['image']['width_px']
    px_per_sec = strip_w / dur

    # nearest-onset time lookup (paper-exact style, same as alignment_metrics)
    sort_idx = np.argsort(gt_strip_x)
    sx_sorted = gt_strip_x[sort_idx].astype(np.float64)
    so_sorted = onset_sec[sort_idx].astype(np.float64)
    nn_idx = np.clip(np.searchsorted(sx_sorted, pred_at_onset), 0, len(sx_sorted) - 1)
    nn_left = np.maximum(nn_idx - 1, 0)
    pick_left = np.abs(sx_sorted[nn_left] - pred_at_onset) < np.abs(sx_sorted[nn_idx] - pred_at_onset)
    nn_idx = np.where(pick_left, nn_left, nn_idx)
    pred_onset_sec = so_sorted[nn_idx]
    err_sec = np.abs(pred_onset_sec - onset_sec)

    # repeat-ambiguity flag per onset
    col_alt = build_repeat_alt_cols(onset_sec, midi_pitch, gt_strip_x, w_scale, fps=fps)
    cols = np.round(gt_strip_x / w_scale).astype(np.int64)
    is_repeat = np.array([int(c) in col_alt for c in cols])

    # local onset density (notes/sec in a +-2s window)
    density = np.zeros(len(onset_sec))
    for i, t0 in enumerate(onset_sec):
        density[i] = np.sum(np.abs(onset_sec - t0) <= 2.0) / 4.0

    rel_pos = onset_sec / max(dur, 1e-6)

    return dict(piece_id=pid, onset_sec=onset_sec, err_sec=err_sec, is_repeat=is_repeat,
               density=density, rel_pos=rel_pos, dur=dur)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    names = ['v13', 'v14', 'v15']
    loaded = []
    for name in names:
        ckpt, cfg_path = MODEL_REGISTRY[name]
        network, cfg = _load_model(ckpt, cfg_path, device)
        loaded.append((name, network, cfg))
    fps = loaded[0][2].data.fps
    h_strip = loaded[0][2].data.h_strip
    w_scale = loaded[0][2].data.w_scale

    proc = Path('data/MSMD/processed')
    emb = Path('data/MSMD/mert_emb')

    out_path = Path('results/per_onset_diagnostic.json')
    out_path.parent.mkdir(exist_ok=True)

    def _dump(results):
        with open(out_path, 'w') as f:
            json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                      for r in results], f)

    all_results = []
    for pid in WORST_SHARED_PIECES:
        try:
            r = run_piece(pid, loaded, proc, emb, device, w_scale, h_strip, fps)
            all_results.append(r)
            print(f'{pid[:50]:50s}  n_onsets={len(r["err_sec"]):4d}  mean_err={r["err_sec"].mean():.2f}s', flush=True)
            _dump(all_results)  # incremental save -- survives a mid-run timeout/kill
        except Exception as e:
            print(f'{pid}: ERROR {e!r}', flush=True)

    print(f'Saved per-onset arrays to {out_path}', flush=True)
    analyze(all_results)


def analyze(all_results):
    """Aggregate analysis -- callable standalone on whatever pieces are
    available (e.g. a partial JSON from a run killed mid-way), not just at
    the end of a full main() run."""
    if not all_results:
        print('No results to analyze.')
        return
    all_err = np.concatenate([np.asarray(r['err_sec']) for r in all_results])
    all_repeat = np.concatenate([np.asarray(r['is_repeat']) for r in all_results])
    all_density = np.concatenate([np.asarray(r['density']) for r in all_results])
    all_relpos = np.concatenate([np.asarray(r['rel_pos']) for r in all_results])

    print(f'\n=== Aggregate over {len(all_err)} onsets across {len(all_results)} pieces ===')
    print(f'mean err (repeat-ambiguous onsets):     {all_err[all_repeat].mean():.3f}s  (n={all_repeat.sum()})')
    print(f'mean err (non-repeat onsets):            {all_err[~all_repeat].mean():.3f}s  (n={(~all_repeat).sum()})')

    dens_median = np.median(all_density)
    sparse = all_density < dens_median
    print(f'mean err (sparse, density<median={dens_median:.2f}): {all_err[sparse].mean():.3f}s  (n={sparse.sum()})')
    print(f'mean err (dense, density>=median):       {all_err[~sparse].mean():.3f}s  (n={(~sparse).sum()})')

    print(f'\ncorrelation(err, is_repeat):  {np.corrcoef(all_err, all_repeat.astype(float))[0,1]:.3f}')
    print(f'correlation(err, density):    {np.corrcoef(all_err, all_density)[0,1]:.3f}')
    print(f'correlation(err, rel_pos):    {np.corrcoef(all_err, all_relpos)[0,1]:.3f}')

    # where in the piece do the WORST onsets (top 10%) cluster?
    thresh = np.percentile(all_err, 90)
    worst_relpos = all_relpos[all_err >= thresh]
    print(f'\ntop-10% worst onsets (err>={thresh:.2f}s): relpos mean={worst_relpos.mean():.2f} '
          f'(0=piece start, 1=piece end), median={np.median(worst_relpos):.2f}')
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        frac = np.mean((worst_relpos >= lo) & (worst_relpos < hi))
        print(f'  relpos [{lo:.1f},{hi:.1f}): {frac*100:.1f}% of worst onsets')


if __name__ == '__main__':
    import sys
    if '--analyze_only' in sys.argv:
        json_path = Path('results/per_onset_diagnostic.json')
        results = json.load(open(json_path))
        print(f'Loaded {len(results)} pieces from {json_path}')
        analyze(results)
    else:
        main()
