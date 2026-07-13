"""F3 -- extends F2's zero-retrain heatmap ensemble several ways at once:

1. Optional 4th member: v13_midi_privileged (E2/E3's checkpoint) shares
   identical strip geometry (h_strip=128, w_scale=4, fps=20) with v13/v14/v15
   even though it's from an early, under-converged training attempt (interim
   solo eval: 49.5% pct@0.5s at ~epoch 5-6). Testing whether folding it in
   (down-weighted) helps or hurts the F2 ensemble.

2. Offline-DTW / particle-filter decode on the ensemble-averaged column
   marginals, on top of the existing threshold+CoM decode -- all computed
   from the SAME forward pass per model-set (collect marginals once, decode
   N ways), mirroring v13_mert_unet/eval_particle_filter.py's E1 pattern. E1
   found DTW decode hurt pct@0.5s but helped mean/median error for a
   converged SINGLE model; testing whether that trade still holds once
   heatmaps are pre-averaged across models (which should already reduce
   per-frame noise).

3. Fusion method: mean (F2's original), median (robust to one member being
   confidently wrong on a frame), or max (confident-vote -- if any member is
   sure, trust it).

4. Pairwise subsets (e.g. --models v13,v14) to check which member pairs
   carry the ensemble's gain, diagnostic for where variance reduction comes
   from.

    python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15 --split test
    python -m mymodel.f3_ensemble_decode.eval --models v13,v14,v15,v13_midi --weights 0.3,0.3,0.3,0.1 --split test
    python -m mymodel.f3_ensemble_decode.eval --models v13,v14 --fusion median --split test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from mymodel.v9_cpjku.cpjku_network import ConditionalUNet
from mymodel.v13_mert_unet.eval import _build_perf_frame
from mymodel.v11_cpjku_fullstrip.data import load_strip_scaled
from mymodel.shared.metrics import alignment_metrics, henkel_metrics
from mymodel.d1_align_matrix.dtw import dtw_decode
from extensions.decode.particle_filter import ParticleFilterXTracker
from mymodel.v13_midi_privileged.repeat_gt import build_repeat_alt_cols
from mymodel.g1_repeat_gnn.infer import load_gnn, build_gnn_alt_cols

MODEL_REGISTRY = {
    'v13':      ('/scratch/pmohseni/results/v13_mert_linear/best_model.pt', 'configs/v13_mert_linear.yaml'),
    'v14':      ('/scratch/pmohseni/results/v14_mert_bilstm/best_model.pt', 'configs/v14_mert_bilstm.yaml'),
    'v15':      ('/scratch/pmohseni/results/v15_mert_mlp/best_model.pt', 'configs/v15_mert_mlp.yaml'),
    'v13_midi': ('/scratch/pmohseni/results/v13_midi_privileged/best_model.pt', 'configs/v13_midi_privileged.yaml'),
}


def _load_model(checkpoint: str, cfg_path: str, device: str):
    cfg = OmegaConf.load(cfg_path)
    net_config = OmegaConf.to_container(cfg.net)
    network = ConditionalUNet(net_config)
    sd = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network.load_state_dict(sd['state_dict'], strict=True)
    network = network.to(device).eval()
    return network, cfg


def _predict_x_com_from_heatmap(heatmap_hw: np.ndarray, threshold: float = 0.5) -> float:
    arr = (heatmap_hw >= threshold).astype(np.float32)
    col = arr.sum(axis=0)
    total = col.sum()
    if total < 1e-6:
        return float(arr.shape[1] // 2)
    xs = np.arange(arr.shape[1], dtype=np.float32)
    return float((xs * col).sum() / total)


def _decode_offline_dtw(marginals: np.ndarray, band_frac: float = 0.05) -> np.ndarray:
    return dtw_decode(marginals.astype(np.float64), band_frac=band_frac).astype(np.float64)


def _decode_particle_filter(marginals: np.ndarray, process_noise_std: float = 3.0, init_std: float = 2.0) -> np.ndarray:
    tracker = ParticleFilterXTracker(process_noise_std=process_noise_std, init_std=init_std)
    T = marginals.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        out[t] = tracker.step(marginals[t])
    return out


def _decode_confidence_velocity(marginals: np.ndarray, conf_window: int = 5,
                                vel_alpha: float = 0.3, conf_gate: float = 0.3) -> np.ndarray:
    """F5 -- confidence-gated velocity-prior decode.

    Per frame, blend the audio observation (local center-of-mass around the
    marginal's peak) with a temporal prior (previous position + velocity),
    weighted by how CONCENTRATED the marginal is around its peak:

        confidence c = (mass within +-conf_window of argmax) / total mass
        x[t] = c * x_obs + (1 - c) * (x[t-1] + v)

    Both failure clusters produce a low-concentration marginal -- repeat
    ambiguity (mass split across two peaks) and sparse audio (flat marginal
    with no dominant peak) -- so both automatically defer to the velocity
    prior, while a sharp unimodal peak (well-tracked frames) yields c~=1 and
    reduces to the raw decode (no precision loss on easy pieces).

    Velocity v is an EMA of confident frame-to-frame steps, seeded from the
    global average scroll rate (W-1)/(T-1) so that even a run of zero
    confident frames (e.g. a long Satie sustain) extrapolates at roughly the
    right tempo instead of stalling.

    conf_gate acts as a FLOOR: below it the marginal is treated as pure noise
    (a flat sustain's argmax is meaningless and would otherwise slowly drag
    the estimate toward column 0), so the effective blend weight is
    rescaled c_eff = max(0, (c - gate) / (1 - gate)). Above the gate the
    blend is smooth; below it the frame is pure velocity extrapolation, and
    velocity is only updated on above-gate frames so noise never corrupts it.
    """
    eps = 1e-6
    T, W = marginals.shape
    x_est = np.zeros(T, dtype=np.float64)
    avg_vel = (W - 1) / max(T - 1, 1)

    def local_com(m):
        tot = m.sum()
        if tot < eps:
            return None, 0.0
        peak = int(np.argmax(m))
        lo, hi = max(0, peak - conf_window), min(W, peak + conf_window + 1)
        local_mass = m[lo:hi].sum()
        xs = np.arange(lo, hi, dtype=np.float64)
        x_obs = float((xs * m[lo:hi]).sum() / local_mass)
        return x_obs, float(local_mass / tot)

    x0, c0 = local_com(marginals[0])
    x_est[0] = x0 if x0 is not None else 0.0
    v = avg_vel
    denom = max(1.0 - conf_gate, eps)
    last_conf_t, last_conf_x = (0, x_est[0]) if c0 >= conf_gate else (None, None)
    for t in range(1, T):
        x_obs, c = local_com(marginals[t])
        x_prior = x_est[t - 1] + v
        c_eff = max(0.0, (c - conf_gate) / denom)
        if x_obs is None or c_eff <= 0.0:
            x_est[t] = x_prior
        else:
            x_est[t] = c_eff * x_obs + (1.0 - c_eff) * x_prior
        if c >= conf_gate:
            # Velocity from spacing between CONSECUTIVE confident observations,
            # divided by the frame gap -- so a big one-frame correction after a
            # sparse gap is NOT misread as high velocity (which would explode).
            if last_conf_t is not None and t > last_conf_t:
                v_obs = (x_obs - last_conf_x) / (t - last_conf_t)
                v = vel_alpha * v_obs + (1.0 - vel_alpha) * v
            last_conf_t, last_conf_x = t, x_obs
    return x_est


def _build_repeat_group_lookup(col_alternates: dict, radius: float):
    """col_alternates: {col: [alt_cols]} from build_repeat_alt_cols -- a
    repeat-equivalence graph derived purely from the score's own pitch-interval
    structure (transposition-invariant n-gram matching), no ground-truth
    position or audio info involved. Same information a performer reading the
    printed score would have (repeat signs, da capo, recurring phrases).

    Returns a function query_col -> [candidate cols] (the repeat GROUP
    containing the nearest known repeat-ambiguous column within `radius`), or
    None if no repeat structure exists near query_col."""
    if not col_alternates:
        return None
    keys = np.array(sorted(col_alternates.keys()), dtype=np.int64)

    def nearest_group(query_col):
        idx = np.searchsorted(keys, query_col)
        cand_idx = [i for i in (idx - 1, idx) if 0 <= i < len(keys)]
        if not cand_idx:
            return None
        best_key = min((int(keys[i]) for i in cand_idx), key=lambda k: abs(k - query_col))
        if abs(best_key - query_col) > radius:
            return None
        return [best_key] + col_alternates[best_key]

    return nearest_group


def _decode_repeat_graph_snap(base_path: np.ndarray, col_alternates: dict, radius: float = 8.0,
                              window: int = 5) -> np.ndarray:
    """base_path: (T,) an already-decoded trajectory in W_sc column space
    (e.g. hybrid_snap's output -- this is a REFINEMENT stacked on top, not a
    replacement, so it inherits F4's already-confirmed precision/drift-rescue
    trade rather than re-deriving an observation estimator from scratch --
    the mistake that sank F5).

    At each frame within `radius` columns of a KNOWN repeat-ambiguous score
    location, snap to whichever member of that repeat group is closest to
    the MEDIAN of the last `window` RAW base-path frames (not the corrected
    output -- comparing against the corrected output creates a self-
    reinforcing absorbing state: once locked onto one repeat instance it can
    never recognize a genuine, sustained transition to the other, since
    "closest to prev" always favors whichever instance was chosen last, even
    against overwhelming contrary evidence. Using a windowed vote over the
    untouched base path instead means an isolated single-frame flicker gets
    corrected, while several consecutive frames genuinely agreeing on a
    transition are still allowed to flip -- at the cost of `window` frames of
    recognition lag, an honest trade rather than permanent lock-in.

    No windowed observation estimator, no velocity model -- unlike F5, this
    never touches frames outside a known repeat-equivalence group, and never
    invents a position the base decoder didn't already produce for some
    frame (it only picks AMONG graph-confirmed candidates)."""
    lookup = _build_repeat_group_lookup(col_alternates, radius)
    out = base_path.copy()
    if lookup is None:
        return out
    T = len(base_path)
    for t in range(T):
        group = lookup(base_path[t])
        if group is None:
            continue
        lo = max(0, t - window)
        recent = base_path[lo:t]
        if len(recent) == 0:
            continue
        ref = float(np.median(recent))
        out[t] = float(min(group, key=lambda c: abs(c - ref)))
    return out


def _parse_cv_name(dname: str):
    """Parse 'cv_w5_g0.3' -> (conf_window=5, conf_gate=0.3)."""
    body = dname[len('cv_'):]
    w_part, g_part = body.split('_g')
    return int(w_part[1:]), float(g_part)


def _fuse(heatmaps_list, weights, fusion: str) -> np.ndarray:
    """heatmaps_list: list of (H, W_sc) arrays, one per model, for a single frame."""
    if fusion == 'mean':
        out = np.zeros_like(heatmaps_list[0])
        for w, hm in zip(weights, heatmaps_list):
            out += w * hm
        return out
    stacked = np.stack(heatmaps_list, axis=0)  # (M, H, W_sc)
    if fusion == 'median':
        return np.median(stacked, axis=0)
    if fusion == 'max':
        return np.max(stacked, axis=0)
    raise ValueError(f'unknown fusion: {fusion}')


@torch.no_grad()
def eval_split(names, weights, processed_root: str, mert_emb_root: str, split: str,
              decoders=('original', 'offline_dtw'), fusion: str = 'mean', dtw_band_frac: float = 0.05,
              pf_process_noise_std: float = 3.0, pf_init_std: float = 2.0, snap_frac: float = 0.1,
              cv_vel_alpha: float = 0.3, repeat_graph_radius: float = 8.0, repeat_graph_window: int = 5,
              gnn_checkpoint: str = None, gnn_sim_threshold: float = 0.85,
              out_dir: str = None, limit: int = None, device: str = None) -> dict | None:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = load_gnn(gnn_checkpoint, device='cpu') if gnn_checkpoint else None
    weights = weights or [1.0 / len(names)] * len(names)
    assert abs(sum(weights) - 1.0) < 1e-6, 'ensemble weights must sum to 1'
    assert len(weights) == len(names)

    loaded = []
    for name in names:
        ckpt, cfg_path = MODEL_REGISTRY[name]
        network, cfg = _load_model(ckpt, cfg_path, device)
        loaded.append((name, network, cfg))
        print(f'Loaded {name} from {ckpt}', flush=True)

    h_strips = {cfg.data.h_strip for _, _, cfg in loaded}
    w_scales = {cfg.data.w_scale for _, _, cfg in loaded}
    fps_set = {cfg.data.fps for _, _, cfg in loaded}
    assert len(h_strips) == 1 and len(w_scales) == 1 and len(fps_set) == 1, \
        f'FAIL: ensemble members have mismatched geometry: h_strip={h_strips} w_scale={w_scales} fps={fps_set}'
    fps = loaded[0][2].data.fps
    h_strip = loaded[0][2].data.h_strip
    w_scale = loaded[0][2].data.w_scale

    proc = Path(processed_root)
    emb = Path(mert_emb_root)
    piece_ids = json.load(open(proc / 'splits.json'))[split]
    if limit:
        piece_ids = piece_ids[:limit]

    out_root = Path(out_dir or f'results/f3_ensemble_decode/{"+".join(names)}') / split
    out_root.mkdir(parents=True, exist_ok=True)

    rows = {d: [] for d in decoders}
    with open(out_root / 'per_piece.jsonl', 'w') as fout:
        for k, pid in enumerate(piece_ids):
            try:
                piece_dir = proc / pid
                ann = json.load(open(piece_dir / 'annotations.json'))
                notes = np.load(piece_dir / 'noteheads.npz')

                strip_w = ann['image']['width_px']
                dur = float(ann['audio']['duration_sec'])
                px_per_sec = strip_w / dur

                feats = np.load(str(emb / f'{pid}.npy')).astype(np.float32)
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
                    member_heatmaps = []
                    for i, (name, network, cfg) in enumerate(loaded):
                        n_frames = cfg.data.n_frames
                        perf_np = _build_perf_frame(feats, t, n_frames)
                        perf_t = torch.from_numpy(perf_np).to(device).unsqueeze(0)
                        out = network(score=score_1, perf=perf_t, hidden=hiddens[i])
                        seg = out['segmentation']
                        new_hidden = out.get('hidden')
                        if new_hidden is not None:
                            hiddens[i] = (new_hidden[0].detach(), new_hidden[1].detach())
                        member_heatmaps.append(seg.squeeze(0).squeeze(0).cpu().numpy())
                    fused_heatmap = _fuse(member_heatmaps, weights, fusion)
                    pred_x_sc_orig[t] = _predict_x_com_from_heatmap(fused_heatmap)
                    marginals[t] = fused_heatmap.sum(axis=0)

                pred_x_sc = {'original': pred_x_sc_orig}
                if 'offline_dtw' in decoders:
                    pred_x_sc['offline_dtw'] = _decode_offline_dtw(marginals, dtw_band_frac)
                if 'particle_filter' in decoders:
                    pred_x_sc['particle_filter'] = _decode_particle_filter(marginals, pf_process_noise_std, pf_init_std)
                if 'hybrid_snap' in decoders:
                    dtw_path = pred_x_sc.get('offline_dtw')
                    if dtw_path is None:
                        dtw_path = _decode_offline_dtw(marginals, dtw_band_frac)
                    diff = np.abs(pred_x_sc_orig - dtw_path)
                    pred_x_sc['hybrid_snap'] = np.where(diff > snap_frac * W_sc, dtw_path, pred_x_sc_orig)
                # F5 confidence-velocity decode: any decoder named cv_w<window>_g<gate>
                # is decoded from the SAME marginals in one pass (grid sweep, cheap).
                for dname in decoders:
                    if dname.startswith('cv_'):
                        w, g = _parse_cv_name(dname)
                        pred_x_sc[dname] = _decode_confidence_velocity(
                            marginals, conf_window=w, vel_alpha=cv_vel_alpha, conf_gate=g)
                if 'repeat_graph_snap' in decoders:
                    base = pred_x_sc.get('hybrid_snap')
                    if base is None:
                        dtw_path = pred_x_sc.get('offline_dtw') or _decode_offline_dtw(marginals, dtw_band_frac)
                        diff = np.abs(pred_x_sc_orig - dtw_path)
                        base = np.where(diff > snap_frac * W_sc, dtw_path, pred_x_sc_orig)
                    col_alt = build_repeat_alt_cols(notes['onset_sec'], notes['midi_pitch'],
                                                    notes['strip_x'], w_scale, fps=fps)
                    pred_x_sc['repeat_graph_snap'] = _decode_repeat_graph_snap(
                        base, col_alt, radius=repeat_graph_radius, window=repeat_graph_window)
                if 'gnn_repeat_snap' in decoders:
                    assert gnn_model is not None, 'gnn_repeat_snap requires --gnn_checkpoint'
                    base = pred_x_sc.get('hybrid_snap')
                    if base is None:
                        dtw_path = pred_x_sc.get('offline_dtw') or _decode_offline_dtw(marginals, dtw_band_frac)
                        diff = np.abs(pred_x_sc_orig - dtw_path)
                        base = np.where(diff > snap_frac * W_sc, dtw_path, pred_x_sc_orig)
                    gnn_col_alt = build_gnn_alt_cols(notes['onset_sec'], notes['midi_pitch'],
                                                     notes['strip_x'], notes.get('measure_idx'),
                                                     w_scale, gnn_model, sim_threshold=gnn_sim_threshold,
                                                     device='cpu')
                    pred_x_sc['gnn_repeat_snap'] = _decode_repeat_graph_snap(
                        base, gnn_col_alt, radius=repeat_graph_radius, window=repeat_graph_window)

                gt_onset = notes['onset_sec']
                gt_strip_x = notes['strip_x']
                frame_idx = np.clip(np.round(gt_onset * fps).astype(int), 0, T - 1)

                per_decoder_m = {}
                for dname in decoders:
                    pred_x_orig = pred_x_sc[dname] * w_scale
                    pred_at_onset = pred_x_orig[frame_idx]
                    m = alignment_metrics(pred_at_onset, gt_strip_x, px_per_sec,
                                          gt_onset_sec=gt_onset, all_strip_x=gt_strip_x,
                                          all_onset_sec=gt_onset)
                    m.update(henkel_metrics(pred_at_onset, gt_strip_x))
                    m['piece_id'] = pid
                    per_decoder_m[dname] = m
                    rows[dname].append(m)

                fout.write(json.dumps({'piece_id': pid, **per_decoder_m}) + '\n')
                fout.flush()

            except Exception as e:
                import traceback
                err = {'piece_id': pid, 'error': repr(e), 'tb': traceback.format_exc()}
                for dname in decoders:
                    rows[dname].append(err)
                fout.write(json.dumps(err) + '\n')
                fout.flush()

            if (k + 1) % 10 == 0:
                line = f'  [{k+1}/{len(piece_ids)}]'
                for dname in decoders:
                    good = [r for r in rows[dname] if 'error' not in r]
                    pct = np.mean([r.get('pct_within_0.5s', 0) for r in good]) if good else float('nan')
                    line += f'  {dname}={pct:.1f}%'
                print(line, flush=True)

    summary = {}
    for dname in decoders:
        good = [r for r in rows[dname] if 'error' not in r]
        if not good:
            print(f'{dname}: ALL PIECES ERRORED, first error: {[r for r in rows[dname] if "error" in r][:1]}')
            continue
        keys = [k for k in good[0] if k.startswith(('mean_', 'median_', 'pct_', 'n'))]
        summ = {'n_pieces': len(good), 'n_errors': len(rows[dname]) - len(good), 'members': names, 'weights': weights}
        for k in keys:
            vals = np.asarray([r[k] for r in good if isinstance(r.get(k), (int, float))])
            if len(vals):
                summ[f'mean_{k}'] = float(vals.mean())
        summ['split'] = split
        summary[dname] = summ

        with open(out_root / f'summary_{dname}.json', 'w') as f:
            json.dump(summ, f, indent=2)

        print(f'\n=== F3 ({"+".join(names)}) decoder={dname} on {split} ===')
        for thr in [0.05, 0.1, 0.5, 1.0, 5.0]:
            kk = f'mean_pct_within_{thr}s'
            if kk in summ:
                print(f'  pct@{thr}s  = {summ[kk]:.1f}%')
        print(f'  mean_err  = {summ.get("mean_mean_abs_err_sec", float("nan")):.3f}s')
        print(f'  median_err= {summ.get("mean_median_abs_err_sec", float("nan")):.3f}s')
        print(f'  n_pieces  = {summ["n_pieces"]}')

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', default='v13,v14,v15', help='comma-separated names from MODEL_REGISTRY')
    p.add_argument('--weights', default=None, help='comma-separated weights, must sum to 1 (default: uniform)')
    p.add_argument('--decoders', default='original,offline_dtw', help='comma-separated: original,offline_dtw,particle_filter,hybrid_snap')
    p.add_argument('--fusion', default='mean', choices=['mean', 'median', 'max'])
    p.add_argument('--dtw_band_frac', type=float, default=0.05)
    p.add_argument('--pf_process_noise_std', type=float, default=3.0)
    p.add_argument('--pf_init_std', type=float, default=2.0)
    p.add_argument('--snap_frac', type=float, default=0.1, help='hybrid_snap: fraction of W_sc disagreement (vs offline_dtw) that triggers a snap')
    p.add_argument('--cv_vel_alpha', type=float, default=0.3, help='confidence-velocity (cv_w<W>_g<G>): EMA weight for velocity updates')
    p.add_argument('--repeat_graph_radius', type=float, default=8.0, help='repeat_graph_snap: max column distance (W_sc space) to a known repeat-ambiguous column')
    p.add_argument('--repeat_graph_window', type=int, default=5, help='repeat_graph_snap: frames of raw-path history to majority-vote over before snapping')
    p.add_argument('--gnn_checkpoint', default=None, help='gnn_repeat_snap: path to G1 trained GCN checkpoint (best_model.pt)')
    p.add_argument('--gnn_sim_threshold', type=float, default=0.85, help='gnn_repeat_snap: cosine-similarity threshold for candidate repeat notes')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--mert_emb_root', default='data/MSMD/mert_emb')
    p.add_argument('--out_dir', default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--device', default=None)
    a = p.parse_args()
    names = a.models.split(',')
    weights = [float(w) for w in a.weights.split(',')] if a.weights else None
    decoders = tuple(a.decoders.split(','))
    eval_split(names, weights, a.processed, a.mert_emb_root, a.split,
              decoders=decoders, fusion=a.fusion, dtw_band_frac=a.dtw_band_frac,
              pf_process_noise_std=a.pf_process_noise_std, pf_init_std=a.pf_init_std,
              snap_frac=a.snap_frac, cv_vel_alpha=a.cv_vel_alpha,
              repeat_graph_radius=a.repeat_graph_radius, repeat_graph_window=a.repeat_graph_window,
              gnn_checkpoint=a.gnn_checkpoint, gnn_sim_threshold=a.gnn_sim_threshold,
              out_dir=a.out_dir, limit=a.limit, device=a.device)


if __name__ == '__main__':
    main()
