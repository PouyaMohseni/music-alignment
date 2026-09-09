"""Globally normalized training: score PATHS, not frames.

THE LABEL BIAS ARGUMENT
-----------------------
The scorer is locally normalized: a softmax over the candidates of one frame,
independently per frame. That forces the probability mass to sum to one AT
EVERY FRAME regardless of how bad every candidate is, so the model structurally
cannot express "I am in a bad state" -- it must still name a favourite. This is
the classic label-bias problem that motivated globally normalized transition
systems, and it explains an otherwise odd pair of measurements: lostness IS
visible in raw objectness (AUC 0.805) yet no mechanism driven by the model's
own score could act on it. Local normalization discards exactly that signal.

A globally normalized model scores a whole path as a sum of UNNORMALIZED terms
and normalizes over paths, so a frame where everything is bad simply
contributes a low score instead of a confident pick.

    path score  = sum_t [ emit(cand_t) + trans(cand_{t-1}, cand_t) ]
    loss        = -score(gold path) + logZ        (forward algorithm)

This matters here because the deployed failure is sequential: 4.65 of the 6.15
remaining points is error propagation, and a per-frame objective never sees a
trajectory at all. It is also the one idea on the list that changes the
OBJECTIVE rather than the decoder, which is where the last two days of decoder
changes all failed.

K is capped for training because the forward algorithm is O(K^2) per
transition; the oracle at K=32 is 95.01 on room, so the cap costs little of the
signal and inference still sees all 256.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.train_cand_scorer import TH, load_dumps, oracle_idx
from extensions.heads.cand_features import NF, build
from extensions.heads.cand_scorer import CandScorer, save

REF, FWD, SIG, JUMP = 5.0, 6.0, 18.0, -6.0


def piece_tensors(p, K, nf, featdim, use_abs_obj):
    """Per-frame features, transitions and the gold index, for one piece."""
    F_, X_, G_, FV_ = [], [], [], []
    x_prev = y_prev = x_prev2 = f_prev = f_prev2 = None
    for i, c in enumerate(p['cand']):
        if c.shape[0] == 0:
            continue
        cs = c[:K]
        fr = int(p['frame'][i])
        dfr = fr - f_prev if f_prev is not None and fr > f_prev else None
        dfp = (f_prev - f_prev2 if f_prev is not None and f_prev2 is not None
               and f_prev > f_prev2 else None)
        F_.append(build(cs, p['bar'][i], p['sys'][i], x_prev, y_prev, dfr,
                        ntot=int(p['ntot'][i]), use_abs_obj=use_abs_obj,
                        x_prev2=x_prev2, dframes_prev=dfp)[:, :nf])
        X_.append(cs[:, 0].astype(np.float32))
        G_.append(oracle_idx(cs, p['t_gt'][i]))
        if featdim:
            fv = p['feat'][i] if p['feat'] is not None else np.zeros((0, featdim), np.float16)
            if fv.shape[0] < cs.shape[0]:
                fv = np.vstack([fv, np.zeros((cs.shape[0] - fv.shape[0], featdim), fv.dtype)])
            FV_.append(fv[:cs.shape[0]].astype(np.float32))
        # the GOLD history advances the features, so the path is the one a
        # correct tracker would have walked
        b = G_[-1]
        x_prev2, f_prev2 = x_prev, f_prev
        x_prev, y_prev, f_prev = float(cs[b, 0]), float(cs[b, 1]), fr
    return F_, X_, G_, FV_


def pad(seq, K, dim):
    out = np.zeros((len(seq), K, dim), np.float32)
    msk = np.zeros((len(seq), K), bool)
    for i, a in enumerate(seq):
        out[i, :a.shape[0]] = a
        msk[i, :a.shape[0]] = True
    return torch.from_numpy(out), torch.from_numpy(msk)


def crf_loss(emit, mask, xs, gold, lam=1.0):
    """emit (T,K) unnormalized; xs (T,K) positions; gold (T,) indices."""
    T, K = emit.shape
    NEG = -1e9
    a = torch.where(mask[0], emit[0], torch.full_like(emit[0], NEG))
    gold_s = emit[0, gold[0]]
    for t in range(1, T):
        d = xs[t].unsqueeze(0) - xs[t - 1].unsqueeze(1)          # (K_prev, K_cur)
        tr = lam * torch.clamp(-0.5 * ((d - FWD) / SIG) ** 2, min=JUMP)
        sc = a.unsqueeze(1) + tr + emit[t].unsqueeze(0)
        sc = torch.where(mask[t].unsqueeze(0), sc, torch.full_like(sc, NEG))
        a = torch.logsumexp(sc, dim=0)
        gold_s = gold_s + emit[t, gold[t]] + tr[gold[t - 1], gold[t]]
    return torch.logsumexp(a, 0) - gold_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', required=True)
    ap.add_argument('--valid', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--K', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--embed', type=int, default=32)
    ap.add_argument('--featproj', type=int, default=8)
    ap.add_argument('--max_frames', type=int, default=120,
                    help='chunk long pieces; the forward pass is sequential')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    torch.set_num_threads(int(__import__('os').environ.get('OMP_NUM_THREADS', '4')))

    tr = load_dumps(sorted(sum([glob.glob(p) for p in a.train], [])))
    va = load_dumps(sorted(sum([glob.glob(p) for p in a.valid], [])))
    fdim = 128 if tr[0]['feat'] is not None else 0
    model = CandScorer(nf=NF, hidden=a.hidden, embed=a.embed, use_abs_obj=True,
                       zdim=0, featdim=fdim, featproj=a.featproj)
    print(f'model: {model.n_params} parameters, globally normalized, K={a.K}',
          flush=True)

    print('building sequences...', flush=True)
    t0 = time.time()
    seqs = []
    for p in tr:
        F_, X_, G_, FV_ = piece_tensors(p, a.K, NF, fdim, True)
        for s in range(0, len(F_), a.max_frames):
            e = s + a.max_frames
            if e - s < 8:
                continue
            f, m = pad(F_[s:e], a.K, NF)
            x, _ = pad([v[:, None] for v in X_[s:e]], a.K, 1)
            fv = pad(FV_[s:e], a.K, fdim)[0] if fdim else None
            seqs.append((f, m, x[:, :, 0], torch.tensor(G_[s:e]), fv))
    print(f'{len(seqs)} chunks in {time.time() - t0:.0f}s', flush=True)

    flat = torch.cat([f[m] for f, m, _, _, _ in seqs[:200]])
    model.set_norm(flat.mean(0).numpy(), flat.std(0).numpy())
    if fdim:
        ff = torch.cat([v[m] for (_, m, _, _, v) in seqs[:200] if v is not None])
        model.set_fnorm(ff.mean(0).numpy(), ff.std(0).numpy())

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for i in rng.permutation(len(seqs)):
            f, m, x, g, fv = seqs[i]
            emit = model(f, m, feat=fv)
            loss = crf_loss(emit, m, x, g) / f.shape[0]
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
        print(f'epoch {ep + 1}  loss {tot / len(seqs):.4f}', flush=True)
        save(model, a.out)
    print(f'wrote {a.out}', flush=True)


if __name__ == '__main__':
    main()
