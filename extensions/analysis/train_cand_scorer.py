"""Fit the candidate selector on the TRAINING split, validate by ROLLOUT.

TWO VALIDATION NUMBERS, AND ONLY ONE OF THEM IS HONEST
------------------------------------------------------
Teacher-forced accuracy asks "given the correct previous position, do you pick
a good candidate?". That is the training objective and it flatters the model,
because at inference the previous position is the model's OWN last choice and
its errors compound. So we also run a greedy ROLLOUT on validation, exactly the
loop the decoder will run, and report hit@0.5s from that. Rollout is the number
to believe; the gap between the two is the exposure bias, and if it is large the
answer is DAgger, not a bigger network.

The reference points printed alongside are the frozen detector's own argmax
(rank 0, i.e. what cyolo_sb does) and the oracle. Any rollout number below the
argmax means the selector is worse than doing nothing.

TEACHER FORCING USES THE PREVIOUS FRAME'S ORACLE-BEST CANDIDATE, not the ground
truth coordinate. The decoder's state is the BOX it last chose, so cloning the
oracle policy means conditioning on a box the detector actually emitted.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

from extensions.heads.cand_features import NF, build
from extensions.heads.cand_scorer import CandScorer, save

TH = 10.0          # 0.5 s at 20 fps, the reported threshold

# Selection and the training target are separate knobs. The soft label
# exp(-|dt|/tau) with tau=3 frames says "land within about 0.15 s", which is
# what the 0.5 s column rewards. The 0.05 s column is a different objective and
# we have never optimised for it: 63.8 there against a causal ceiling of 94.0,
# a 30-point gap next to 4.6 at 0.5 s.


def load_dumps(paths):
    """-> list of pieces, each a dict of parallel per-frame arrays."""
    pieces = []
    for p in paths:
        z = np.load(p, allow_pickle=False)
        names = sorted({k.split('||')[0] for k in z.files})
        for nm in names:
            lens = z[f'{nm}||lens']
            if lens.size == 0:
                continue
            flat = z[f'{nm}||cand']
            off = np.concatenate([[0], np.cumsum(lens)])
            zk, fk = f'{nm}||z', f'{nm}||feat'
            fl = z[f'{nm}||flens'] if f'{nm}||flens' in z.files else None
            if fl is not None:
                foff = np.concatenate([[0], np.cumsum(fl)])
                fflat = z[fk]
                feats = [fflat[foff[i]:foff[i + 1]] for i in range(len(fl))]
            else:
                feats = None
            pieces.append(dict(
                name=nm, frame=z[f'{nm}||frame'], t_gt=z[f'{nm}||t_gt'],
                ntot=z[f'{nm}||ntot'], bar=z[f'{nm}||bar'], sys=z[f'{nm}||sys'],
                z=(z[zk] if zk in z.files else None), feat=feats,
                cand=[flat[off[i]:off[i + 1]] for i in range(len(lens))]))
    return pieces


def index(pieces, skip_first=True):
    """Flat (piece, frame) index. The first scored frame of a piece has no
    previous position, so it carries no displacement signal to learn from."""
    return [(pi, fi) for pi, p in enumerate(pieces)
            for fi in range(1 if skip_first else 0, len(p['cand']))
            if len(p['cand'][fi]) > 0]


def oracle_idx(c, t_gt):
    return int(np.argmin(np.abs(c[:, 5] - t_gt))) if len(c) else -1


def make_batch(pieces, items, rng, noise_p=0.0, noise_px=30.0, use_abs_obj=True,
               featdim=0):
    feats, labels, zs, fvs = [], [], [], []
    for pi, fi in items:
        p = pieces[pi]
        c = p['cand'][fi]
        prev = p['cand'][fi - 1]
        if len(prev):
            b = oracle_idx(prev, p['t_gt'][fi - 1])
            x_prev, y_prev = float(prev[b, 0]), float(prev[b, 1])
            if noise_p > 0 and rng.random() < noise_p:
                # heavy-tailed, so the model sees both a few-pixel drift and the
                # occasional gross mistake it will have to recover from
                x_prev += float(rng.laplace(0.0, noise_px))
        else:
            x_prev = y_prev = None
        dfr = int(p['frame'][fi] - p['frame'][fi - 1]) if p['frame'][fi] >= 0 else None
        feats.append(build(c, p['bar'][fi], p['sys'][fi], x_prev, y_prev, dfr,
                           ntot=int(p['ntot'][fi]), use_abs_obj=use_abs_obj))
        labels.append(np.abs(c[:, 5] - p['t_gt'][fi]))
        zs.append(p['z'][fi] if p['z'] is not None else np.zeros(128, np.float32))
        if featdim:
            fv = p['feat'][fi] if p['feat'] is not None else np.zeros((0, featdim), np.float16)
            # the feature dump is capped tighter than the candidate list, so
            # pad the tail rather than silently misaligning candidate k with
            # feature k of a different candidate
            if fv.shape[0] < c.shape[0]:
                fv = np.vstack([fv, np.zeros((c.shape[0] - fv.shape[0], featdim), fv.dtype)])
            fvs.append(fv[:c.shape[0]].astype(np.float32))
    K = max(f.shape[0] for f in feats)
    B = len(feats)
    X = np.zeros((B, K, NF), np.float32)
    E = np.full((B, K), 1e9, np.float32)
    M = np.zeros((B, K), bool)
    F = np.zeros((B, K, featdim), np.float32) if featdim else None
    for i, (f, e) in enumerate(zip(feats, labels)):
        k = f.shape[0]
        X[i, :k], E[i, :k], M[i, :k] = f, e, True
        if featdim:
            F[i, :k] = fvs[i]
    return (torch.from_numpy(X), torch.from_numpy(E), torch.from_numpy(M),
            torch.from_numpy(np.stack(zs)),
            torch.from_numpy(F) if featdim else None)


@torch.no_grad()
def rollout(model, pieces, use_abs_obj=True, device='cpu', th=TH):
    """Greedy decode, exactly the loop the decoder runs. Returns (rollout hit,
    argmax hit, oracle hit) as percentages over every scored frame."""
    hit = arg = orc = n = 0
    for p in pieces:
        x_prev = y_prev = None
        for fi, c in enumerate(p['cand']):
            if len(c) == 0:
                continue
            n += 1
            err = np.abs(c[:, 5] - p['t_gt'][fi])
            arg += err[0] <= th
            orc += err.min() <= th
            dfr = (int(p['frame'][fi] - p['frame'][fi - 1])
                   if fi > 0 and p['frame'][fi] >= 0 else None)
            f = build(c, p['bar'][fi], p['sys'][fi], x_prev, y_prev, dfr,
                      ntot=int(p['ntot'][fi]), use_abs_obj=use_abs_obj)
            zz = (torch.from_numpy(p['z'][fi]).unsqueeze(0)
                  if p['z'] is not None else None)
            ff = None
            if model.fenc is not None and p['feat'] is not None:
                fv = p['feat'][fi].astype(np.float32)
                if fv.shape[0] < c.shape[0]:
                    fv = np.vstack([fv, np.zeros((c.shape[0] - fv.shape[0],
                                                  model.featdim), np.float32)])
                ff = torch.from_numpy(fv[:c.shape[0]]).unsqueeze(0)
            s = model(torch.from_numpy(f).unsqueeze(0).to(device), z=zz, feat=ff)[0]
            j = int(s.argmax())
            hit += err[j] <= th
            x_prev, y_prev = float(c[j, 0]), float(c[j, 1])
    return (100.0 * hit / max(n, 1), 100.0 * arg / max(n, 1),
            100.0 * orc / max(n, 1), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--valid', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--tau', type=float, default=3.0)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--embed', type=int, default=32)
    ap.add_argument('--noise_p', type=float, default=0.3)
    ap.add_argument('--noise_px', type=float, default=30.0)
    ap.add_argument('--no_abs_obj', action='store_true')
    ap.add_argument('--sel_th', type=float, default=TH,
                    help='rollout threshold in FRAMES used to pick the best '
                         'epoch (10 = 0.5 s, 1 = 0.05 s)')
    ap.add_argument('--use_feat', action='store_true',
                    help='per-candidate backbone features, the 128 numbers the '
                         '1,935-parameter Detect conv maps to objectness')
    ap.add_argument('--use_z', action='store_true',
                    help="give the selector the detector's own audio vector")
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    use_abs = not a.no_abs_obj

    tr = load_dumps(sorted(sum([glob.glob(p) for p in a.train], [])))
    va = load_dumps(sorted(sum([glob.glob(p) for p in a.valid], [])))
    idx = index(tr)
    print(f'train: {len(tr)} pieces, {len(idx)} usable frames', flush=True)
    print(f'valid: {len(va)} pieces, {sum(len(p["cand"]) for p in va)} frames', flush=True)

    # normalisation from a sample of the training features, no noise applied
    samp = [idx[i] for i in rng.choice(len(idx), min(4000, len(idx)), replace=False)]
    fdim = 128 if (a.use_feat and tr[0]['feat'] is not None) else 0
    if a.use_feat and not fdim:
        raise SystemExit('--use_feat but the dump carries no features')
    Xs, _, Ms, Zs, Fs = make_batch(tr, samp, rng, use_abs_obj=use_abs, featdim=fdim)
    flat = Xs[Ms]
    zdim = Zs.shape[1] if (a.use_z and tr[0]['z'] is not None) else 0
    if a.use_z and not zdim:
        raise SystemExit('--use_z but the dump carries no z; re-dump first')
    model = CandScorer(hidden=a.hidden, embed=a.embed, use_abs_obj=use_abs,
                       zdim=zdim, featdim=fdim)
    model.set_norm(flat.mean(0).numpy(), flat.std(0).numpy())
    if zdim:
        model.set_znorm(Zs.mean(0).numpy(), Zs.std(0).numpy())
    if fdim:
        flat_f = Fs[Ms]
        model.set_fnorm(flat_f.mean(0).numpy(), flat_f.std(0).numpy())
    print(f'model: {model.n_params} parameters', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(len(idx))
        tot = nb = 0.0
        for i in range(0, len(order), a.bs):
            items = [idx[j] for j in order[i:i + a.bs]]
            X, E, M, Z, F = make_batch(tr, items, rng, a.noise_p, a.noise_px,
                                       use_abs, featdim=fdim)
            s = model(X, M, z=(Z if model.zenc is not None else None), feat=F)
            with torch.no_grad():                      # soft target on error
                tgt = torch.softmax(torch.where(M, -E / a.tau, torch.full_like(E, -1e9)), -1)
            loss = -(tgt * torch.log_softmax(s, -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        model.eval()
        r, arg, orc, n = rollout(model, va, use_abs, th=a.sel_th)
        flag = ''
        if r > best:
            best, best_state = r, {k: v.clone() for k, v in model.state_dict().items()}
            flag = '  <- best'
        print(f'ep {ep:3d}  loss {tot / nb:.4f}   valid rollout {r:5.2f}   '
              f'(argmax {arg:5.2f}  oracle {orc:5.2f}){flag}', flush=True)

    model.load_state_dict(best_state)
    r, arg, orc, n = rollout(model, va, use_abs, th=a.sel_th)
    r5, a5, o5, _ = rollout(model, va, use_abs, th=TH)
    print(f'  at 0.5 s: rollout {r5:.2f}  argmax {a5:.2f}  oracle {o5:.2f}')
    print(f'\nBEST valid rollout hit@0.5s = {r:.2f}   argmax {arg:.2f}   '
          f'oracle {orc:.2f}   over {n} frames')
    print(f'  selector recovers {100 * (r - arg) / max(orc - arg, 1e-9):.1f}% of the '
          f'available headroom')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save(model, a.out, extra=dict(valid_rollout=r, valid_argmax=arg,
                                  valid_oracle=orc, use_abs_obj=use_abs))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
