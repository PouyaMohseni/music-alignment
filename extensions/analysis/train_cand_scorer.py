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
import time
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')

from extensions.heads.cand_features import NF, build
from extensions.heads.cand_scorer import CandScorer, load, save

TH = 10.0          # 0.5 s at 20 fps, the reported threshold

# --- the hand score, for --residual -----------------------------------------
# Deployment blends the learned and hand scores, beta*s + (1-beta)*h, and beta
# turned out to be the one constant no proxy can choose: on synthetic audio the
# scorer is good enough that beta=1 wins, on the real recordings it is
# overconfident and needs the prior to hold it back. --residual removes the
# knob by training the scorer as a CORRECTION on top of the hand score, so the
# prior always contributes: the fitted quantity is s = score - h, and decoding
# is argmax(s + h), which is the existing blend formula at beta=0.5 up to a
# factor of two. Such a checkpoint must therefore be decoded at beta=0.5.
#
# The prior baked in here becomes part of the model, so it must NOT be the
# 6/18/-6 the project shipped: those constants were swept on the room test
# recordings. Validation, held-out and leave-one-piece-out selection all
# independently pick 10/18/-8, so that is the default, and --prior overrides it.
PRIOR_REF = 5.0
PRIOR = (10.0, 18.0, -8.0)


def hand_score(cand, x_prev, dframes, prior=None):
    """log objectness plus the transition prior, per candidate."""
    fwd, sig, jump = prior or PRIOR
    lo = np.log(np.clip(cand[:, 4], 1e-8, None)).astype(np.float32)
    if x_prev is None:
        return lo
    k = float(np.clip((dframes or PRIOR_REF) / PRIOR_REF, 0.2, 8.0))
    d = cand[:, 0] - float(x_prev)
    return lo + np.maximum(-0.5 * ((d - fwd * k) / sig) ** 2, jump).astype(np.float32)

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


def _pitch(p, fi):
    v = p.get('pitch')
    return None if v is None or fi >= len(v) else v[fi]


def _vhat(p, fi):
    v = p.get('vhat')
    if v is None or fi >= len(v) or not np.isfinite(v[fi]):
        return None
    return float(v[fi])


def oracle_tempo(p, alpha=0.2, vmax=40.0):
    """Causal EMA of the TRUE px/frame, one value per frame.

    Entry i is the estimate available BEFORE frame i is scored, so the feature
    never sees its own answer. At inference the decoder keeps the same EMA over
    its own choices; the estimate is smoothed over the piece, so the gap
    between the two is far milder than for a raw previous position.
    """
    n = len(p['cand'])
    out = np.full(n, np.nan, np.float32)
    v = xp = fp = None
    for i, c in enumerate(p['cand']):
        if v is not None:
            out[i] = v
        if len(c) == 0:
            continue
        j = oracle_idx(c, p['t_gt'][i])
        x, fr = float(c[j, 0]), int(p['frame'][i])
        if xp is not None and fr > fp:
            vo = (x - xp) / float(fr - fp)
            if 0.0 < vo < vmax:
                v = vo if v is None else (1 - alpha) * v + alpha * vo
        xp, fp = x, fr
    return out


def make_batch(pieces, items, rng, noise_p=0.0, noise_px=30.0, use_abs_obj=True,
               featdim=0, states=None, dagger_frac=0.0, prior=None):
    """states maps (piece, frame) -> the history the POLICY reached there.

    Teacher forcing builds x_prev from the oracle, so the model is fitted on a
    history it never sees at inference; noise_px is a guess at the shape of its
    own mistakes. When a rollout state is available and sampled, it replaces
    the oracle history entirely -- and the noise is skipped, because the
    measured error distribution is what the guess was standing in for.
    """
    feats, labels, zs, fvs, hands = [], [], [], [], []
    for pi, fi in items:
        p = pieces[pi]
        c = p['cand'][fi]
        st = None
        if states is not None and dagger_frac > 0 and rng.random() < dagger_frac:
            st = states.get((pi, fi))
        if st is not None:
            x_prev, y_prev, x_prev2, dfr, dfr_prev = st
            feats.append(build(c, p['bar'][fi], p['sys'][fi], x_prev, y_prev, dfr,
                               ntot=int(p['ntot'][fi]), use_abs_obj=use_abs_obj,
                               x_prev2=x_prev2, dframes_prev=dfr_prev,
                               v_hat=_vhat(p, fi), pitch=_pitch(p, fi)))
            hands.append(hand_score(c, x_prev, dfr, prior))
            labels.append(np.abs(c[:, 5] - p['t_gt'][fi]))
            zs.append(p['z'][fi] if p['z'] is not None else np.zeros(128, np.float32))
            if featdim:
                fv = p['feat'][fi] if p['feat'] is not None else np.zeros((0, featdim), np.float16)
                if fv.shape[0] < c.shape[0]:
                    fv = np.vstack([fv, np.zeros((c.shape[0] - fv.shape[0], featdim), fv.dtype)])
                fvs.append(fv[:c.shape[0]].astype(np.float32))
            continue
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
        # a second step back, so the model can see SPEED and not only position
        x_prev2 = dfr_prev = None
        if fi >= 2 and len(p['cand'][fi - 2]):
            pv2 = p['cand'][fi - 2]
            b2 = oracle_idx(pv2, p['t_gt'][fi - 2])
            x_prev2 = float(pv2[b2, 0])
            if p['frame'][fi - 1] >= 0:
                dfr_prev = int(p['frame'][fi - 1] - p['frame'][fi - 2])
        dfr = int(p['frame'][fi] - p['frame'][fi - 1]) if p['frame'][fi] >= 0 else None
        feats.append(build(c, p['bar'][fi], p['sys'][fi], x_prev, y_prev, dfr,
                           ntot=int(p['ntot'][fi]), use_abs_obj=use_abs_obj,
                           x_prev2=x_prev2, dframes_prev=dfr_prev,
                           v_hat=_vhat(p, fi), pitch=_pitch(p, fi)))
        hands.append(hand_score(c, x_prev, dfr, prior))
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
    H = np.zeros((B, K), np.float32)
    F = np.zeros((B, K, featdim), np.float32) if featdim else None
    for i, (f, e) in enumerate(zip(feats, labels)):
        k = f.shape[0]
        X[i, :k], E[i, :k], M[i, :k], H[i, :k] = f, e, True, hands[i]
        if featdim:
            F[i, :k] = fvs[i]
    return (torch.from_numpy(X), torch.from_numpy(E), torch.from_numpy(M),
            torch.from_numpy(np.stack(zs)),
            torch.from_numpy(F) if featdim else None, torch.from_numpy(H))


@torch.no_grad()
def rollout(model, pieces, use_abs_obj=True, device='cpu', th=TH,
            tempo_alpha=0.2, vmax=40.0, residual=False, prior=None):
    """Greedy decode, exactly the loop the decoder runs. Returns (rollout hit,
    argmax hit, oracle hit) as percentages over every scored frame."""
    hit = arg = orc = n = 0
    for p in pieces:
        x_prev = y_prev = x_prev2 = None
        f_prev = f_prev2 = None
        v_hat = None            # tracked from the policy's own steps, as at
        for fi, c in enumerate(p['cand']):   # inference -- never the oracle's
            if len(c) == 0:
                continue
            n += 1
            err = np.abs(c[:, 5] - p['t_gt'][fi])
            arg += err[0] <= th
            orc += err.min() <= th
            dfr = (int(p['frame'][fi] - p['frame'][fi - 1])
                   if fi > 0 and p['frame'][fi] >= 0 else None)
            dfr_prev = (int(f_prev - f_prev2)
                        if f_prev is not None and f_prev2 is not None else None)
            f = build(c, p['bar'][fi], p['sys'][fi], x_prev, y_prev, dfr,
                      ntot=int(p['ntot'][fi]), use_abs_obj=use_abs_obj,
                      x_prev2=x_prev2, dframes_prev=dfr_prev, v_hat=v_hat,
                      pitch=(p['pitch'][fi] if p.get('pitch') is not None else None))
            zz = (torch.from_numpy(p['z'][fi]).unsqueeze(0)
                  if p['z'] is not None else None)
            ff = None
            if model.fenc is not None and p['feat'] is not None:
                fv = p['feat'][fi].astype(np.float32)
                if fv.shape[0] < c.shape[0]:
                    fv = np.vstack([fv, np.zeros((c.shape[0] - fv.shape[0],
                                                  model.featdim), np.float32)])
                ff = torch.from_numpy(fv[:c.shape[0]]).unsqueeze(0)
            s = model(torch.from_numpy(f[:, :model.nf]).unsqueeze(0).to(device),
                      z=zz, feat=ff)[0]
            if residual:
                s = s + torch.from_numpy(hand_score(c, x_prev, dfr, prior)).to(device)
            j = int(s.argmax())
            hit += err[j] <= th
            xn = float(c[j, 0])
            if x_prev is not None and dfr:
                vo = (xn - x_prev) / float(dfr)
                if 0.0 < vo < vmax:            # gated: a lost step must not
                    v_hat = vo if v_hat is None else \
                        (1 - tempo_alpha) * v_hat + tempo_alpha * vo
            x_prev2, f_prev2 = x_prev, f_prev
            x_prev, y_prev = xn, float(c[j, 1])
            f_prev = int(p['frame'][fi])
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
    ap.add_argument('--no_tempo', action='store_true',
                    help='drop the 4 tracked-tempo features (they are last, so '
                         'truncation is exact for them)')
    ap.add_argument('--tempo_alpha', type=float, default=0.2,
                    help='EMA rate for the tracked tempo fed to the features')
    ap.add_argument('--no_vel', action='store_true',
                    help='drop the 4 velocity features, so projection width '
                         'and velocity can be separated in the grid')
    ap.add_argument('--featproj', type=int, default=32,
                    help='width of the per-candidate feature projection')
    ap.add_argument('--use_feat', action='store_true',
                    help='per-candidate backbone features, the 128 numbers the '
                         '1,935-parameter Detect conv maps to objectness')
    ap.add_argument('--use_z', action='store_true',
                    help="give the selector the detector's own audio vector")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prior', default=','.join(str(v) for v in PRIOR),
                    help='fwd,sigma,jump for the hand score baked in by --residual')
    ap.add_argument('--residual', action='store_true',
                    help='fit a correction on top of the hand score, so the prior '
                         'always contributes and no blend weight has to be chosen; '
                         'decode such a checkpoint at blend 0.5')
    ap.add_argument('--dagger_init', default='',
                    help='checkpoint to roll out for DAgger states; its own '
                         'visited histories replace the oracle ones')
    ap.add_argument('--dagger_frac', type=float, default=0.5,
                    help='fraction of training items drawn from rollout states')
    ap.add_argument('--pitch_heads', default='',
                    help='checkpoint of the two pitch heads; their per-candidate '
                         'AGREEMENT becomes features 37..40')
    ap.add_argument('--recover_w', type=float, default=1.0,
                    help='loss weight for states where NO candidate is inside '
                         'the threshold (1.0 = current behaviour, 0 = drop)')
    ap.add_argument('--dagger_rounds', type=int, default=1,
                    help='re-roll with the model being trained after each round')
    a = ap.parse_args()
    prior = tuple(float(v) for v in a.prior.split(','))
    if a.residual:
        print(f'RESIDUAL: fitting a correction on the hand score, prior {prior}',
              flush=True)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    use_abs = not a.no_abs_obj

    tr = load_dumps(sorted(sum([glob.glob(p) for p in a.train], [])))
    va = load_dumps(sorted(sum([glob.glob(p) for p in a.valid], [])))
    if a.pitch_heads:
        from extensions.analysis.pitch_heads import annotate_pieces
        annotate_pieces(tr, a.pitch_heads)
        annotate_pieces(va, a.pitch_heads)
    for _p in tr + va:
        _p['vhat'] = oracle_tempo(_p, alpha=a.tempo_alpha)
    idx = index(tr)
    print(f'train: {len(tr)} pieces, {len(idx)} usable frames', flush=True)
    print(f'valid: {len(va)} pieces, {sum(len(p["cand"]) for p in va)} frames', flush=True)

    # normalisation from a sample of the training features, no noise applied
    samp = [idx[i] for i in rng.choice(len(idx), min(4000, len(idx)), replace=False)]
    fdim = 128 if (a.use_feat and tr[0]['feat'] is not None) else 0
    if a.use_feat and not fdim:
        raise SystemExit('--use_feat but the dump carries no features')
    Xs, _, Ms, Zs, Fs, _ = make_batch(tr, samp, rng, use_abs_obj=use_abs, featdim=fdim)
    # FEATURE_NAMES only ever grows at the end, so truncation is exactly the
    # older feature set rather than an approximation of it
    # --no_vel dropped the 4 velocity features by TRUNCATING to NF-4, which
    # was correct only while velocity was last in FEATURE_NAMES. Nine
    # neighbourhood features and four tempo features have been appended since,
    # so NF-4 now removes TEMPO and keeps velocity -- the opposite of the flag's
    # name, silently. The models it produced (novel_*) all lost anyway, 90.3 to
    # 91.4 against 92.7 to 93.4 with velocity, so rather than reshuffle the
    # feature order to rescue a losing ablation, refuse it.
    if a.no_vel:
        raise SystemExit(
            '--no_vel is expressed as truncation to NF-4, which no longer '
            'selects the velocity block: FEATURE_NAMES has grown past it. '
            'Zero the columns explicitly if this ablation is needed again.')
    nf_use = NF if not a.no_tempo else 33
    Xs = Xs[:, :, :nf_use]
    flat = Xs[Ms]
    zdim = Zs.shape[1] if (a.use_z and tr[0]['z'] is not None) else 0
    if a.use_z and not zdim:
        raise SystemExit('--use_z but the dump carries no z; re-dump first')
    model = CandScorer(nf=nf_use, hidden=a.hidden, embed=a.embed,
                       use_abs_obj=use_abs, zdim=zdim, featdim=fdim,
                       featproj=a.featproj)
    model.set_norm(flat.mean(0).numpy(), flat.std(0).numpy())
    if zdim:
        model.set_znorm(Zs.mean(0).numpy(), Zs.std(0).numpy())
    if fdim:
        flat_f = Fs[Ms]
        model.set_fnorm(flat_f.mean(0).numpy(), flat_f.std(0).numpy())
    print(f'model: {model.n_params} parameters', flush=True)

    # ---- DAgger states ----------------------------------------------------
    # The selector's input depends on its OWN previous position, so teacher
    # forcing fits it on a history it never sees. The measured cost is not
    # subtle: deleting the previous-position noise, the crude stand-in for that
    # history, drops validation rollout from 95.33 to 92.38 -- below the
    # detector's own argmax. Rather than guess the shape of the model's
    # mistakes with a Laplace draw, roll a policy out and use the states it
    # actually reaches. The oracle action is known at every one of them (the
    # candidate nearest ground truth), which is why this is imitation learning
    # and not RL.
    states = None
    if a.dagger_init:
        from extensions.analysis.dagger import rollout_states
        init = load(a.dagger_init)[0] if a.dagger_init != 'self' else model
        # The rollout policy has its OWN feature width, which need not be the
        # trainee's: adding nine features today made nf_use 33 while vel_p8 is
        # still 24, and truncating its input to 33 fed it nine columns it was
        # never fitted with. Truncate to whatever the policy expects.
        t0 = time.time()
        st = rollout_states(init, tr, build, torch, use_abs_obj=use_abs,
                            featdim=fdim, nf=init.nf)
        states = {(pi, fi): (xp, yp, xp2, d, dp) for pi, fi, xp, yp, xp2, d, dp in st}
        vis = sum(1 for v in states.values() if v[0] is not None)
        print(f'DAgger: {len(states)} visited states from '
              f'{a.dagger_init} in {time.time() - t0:.0f}s '
              f'({vis} with a history), frac={a.dagger_frac}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    best, best_state = -1.0, None
    every = max(a.epochs // max(a.dagger_rounds, 1), 1)
    for ep in range(a.epochs):
        # later rounds re-roll with the model being trained, so the states keep
        # matching the policy rather than its ancestor
        if states is not None and a.dagger_rounds > 1 and ep and ep % every == 0:
            from extensions.analysis.dagger import rollout_states
            model.eval()
            st = rollout_states(model, tr, build, torch, use_abs_obj=use_abs,
                                featdim=fdim, nf=model.nf)
            states = {(pi, fi): (xp, yp, xp2, d, dp)
                      for pi, fi, xp, yp, xp2, d, dp in st}
            print(f'  [ep {ep}] re-rolled: {len(states)} states', flush=True)
        model.train()
        order = rng.permutation(len(idx))
        tot = nb = 0.0
        for i in range(0, len(order), a.bs):
            items = [idx[j] for j in order[i:i + a.bs]]
            X, E, M, Z, F, H = make_batch(tr, items, rng, a.noise_p, a.noise_px,
                                          use_abs, featdim=fdim,
                                          states=states, dagger_frac=a.dagger_frac,
                                          prior=prior)
            s = model(X[:, :, :nf_use], M,
                      z=(Z if model.zenc is not None else None), feat=F)
            if a.residual:
                s = s + H          # fit the correction, not the whole score
            with torch.no_grad():                      # soft target on error
                tgt = torch.softmax(torch.where(M, -E / a.tau, torch.full_like(E, -1e9)), -1)
            per = -(tgt * torch.log_softmax(s, -1)).sum(-1)
            if a.recover_w < 1.0:
                # UNWINNABLE STATES. The target is already soft, but at a state
                # where NO candidate is inside the threshold the softmax still
                # concentrates on the least-bad wrong one, so the model is
                # taught to prefer a wrong answer. Teacher forcing almost never
                # visits such states; DAgger's rollout states are full of them,
                # which is the likeliest reason DAgger lost 8 points of
                # validation headroom. Down-weight them instead of pretending
                # the least-bad box is a target.
                with torch.no_grad():
                    # NOT `best`: that name already holds the best validation
                    # score in this function, and `with torch.no_grad()` is not
                    # a scope, so binding it here clobbered a float with a
                    # tensor and `if r > best` then raised.
                    best_err = E.masked_fill(~M, 1e9).min(-1).values
                    w = torch.where(best_err <= a.sel_th,
                                    torch.ones_like(best_err),
                                    torch.full_like(best_err, a.recover_w))
                loss = (per * w).sum() / w.sum().clamp_min(1e-6)
            else:
                loss = per.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        model.eval()
        r, arg, orc, n = rollout(model, va, use_abs, th=a.sel_th, residual=a.residual, prior=prior)
        flag = ''
        if r > best:
            best, best_state = r, {k: v.clone() for k, v in model.state_dict().items()}
            flag = '  <- best'
        print(f'ep {ep:3d}  loss {tot / nb:.4f}   valid rollout {r:5.2f}   '
              f'(argmax {arg:5.2f}  oracle {orc:5.2f}){flag}', flush=True)

    model.load_state_dict(best_state)
    r, arg, orc, n = rollout(model, va, use_abs, th=a.sel_th, residual=a.residual, prior=prior)
    r5, a5, o5, _ = rollout(model, va, use_abs, th=TH, residual=a.residual, prior=prior)
    print(f'  at 0.5 s: rollout {r5:.2f}  argmax {a5:.2f}  oracle {o5:.2f}')
    print(f'\nBEST valid rollout hit@0.5s = {r:.2f}   argmax {arg:.2f}   '
          f'oracle {orc:.2f}   over {n} frames')
    print(f'  selector recovers {100 * (r - arg) / max(orc - arg, 1e-9):.1f}% of the '
          f'available headroom')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save(model, a.out, extra=dict(valid_rollout=r, valid_argmax=arg,
                                  valid_oracle=orc, use_abs_obj=use_abs,
                                  residual=a.residual))
    if a.residual:
        print('RESIDUAL checkpoint: decode at blend 0.5, which is argmax(s + h)')
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
