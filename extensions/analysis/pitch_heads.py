"""Two pitch heads, trained with MIDI, run on image and audio features alone.

WHY. The failure analysis says errors are pitch-DISTINGUISHABLE: median pitch
overlap between the chosen and the true position is 0.000, and 59.2% of errors
land where no notes are shared. An oracle probe confirmed it -- pitch agreement
is 4.70x more likely on the correct candidate than on a wrong one (0.398 vs
0.085), which is far better separated than the lostness signal that killed four
mechanisms (AUC 0.805 but 0.33 precision). Critically it is
HISTORY-INDEPENDENT, which is what the 4.65-point propagation term needs.

THE CONSTRAINT. Pitch may only ever be TRAINING supervision. The method has to
work on music with no symbolic reference, so nothing symbolic may be read at
inference. Hence two heads:

  score head   backbone feat (128) at a candidate -> the MIDI pitch of the
               notehead it sits on. That is readable from the image: a
               notehead's height on the staff IS its pitch.
  audio head   z (128) for a frame -> which pitches are sounding. That is
               readable from the audio.

MIDI supplies both targets during training and is absent at inference.

WHY IT MUST BE AN INTERACTION. An audio pitch estimate is identical for every
candidate in a frame, and a value constant across a frame can rank nothing --
the reason the z vector failed twice as a feature. Only the AGREEMENT between
each candidate's own predicted pitch and the sounding pitches varies per
candidate, and that is what gets fed to the scorer.

Labels are joined in the onset domain: cand[:,5] is the candidate's position
mapped through the piece's x->onset interpolator, so it identifies which
notehead the box sits on BY POSITION. Matching on x instead would compare
unrolled coordinates against raw per-page note_x, which is what made the first
version of the oracle probe report a meaningless 1.14x.
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.offline_decode import TH, load

FPS = 22050.0 / 1102.0
MSMD = '/scratch/pmohseni/datasets/cyolo_data/msmd'
LO, HI = 21, 109                      # piano range, 88 pitches
NP_ = HI - LO


def piece_notes(name):
    stem = name.split('_page_')[0]
    for d in ('msmd_rp', 'msmd_train', 'msmd_valid', 'msmd_test'):
        for f in glob.glob(f'{MSMD}/{d}/{stem}.npz'):
            z = np.load(f, allow_pickle=True)
            o = np.array([[float(c['onset']), int(c['pitch'])] for c in z['coords']])
            return o if len(o) else None
    return None


def load_feat(paths):
    """Pages with per-candidate features and the per-frame audio vector."""
    pages = load(paths)
    for pat in np.atleast_1d(paths):
        for fp in sorted(glob.glob(pat)):
            z = np.load(fp, allow_pickle=False)
            for nm in sorted({k.split('||')[0] for k in z.files}):
                fl = z.get(f'{nm}||flens')
                for pg in pages:
                    if pg['name'] != nm:
                        continue
                    if fl is not None:
                        off = np.concatenate([[0], np.cumsum(fl)])
                        flat = z[f'{nm}||feat']
                        pg['feat'] = [flat[off[i]:off[i + 1]] for i in range(len(fl))]
                    if f'{nm}||z' in z.files:
                        pg['z'] = z[f'{nm}||z']
    return pages


def build_xy(paths, tol=0.05, max_cand=64):
    """-> (feat, pitch_label) for the score head, (z, multi-hot) for the audio head."""
    F, PL, Z, SND = [], [], [], []
    for p in load_feat(paths):
        nt = piece_notes(p['name'])
        if nt is None or p.get('feat') is None or p.get('z') is None:
            continue
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            t_now = float(p['t_gt'][i]) / FPS
            snd = nt[np.abs(nt[:, 0] - t_now) <= tol, 1].astype(int)
            if not len(snd):
                continue
            v = np.zeros(NP_, np.float32)
            v[np.clip(snd - LO, 0, NP_ - 1)] = 1.0
            Z.append(np.asarray(p['z'][i], np.float32)); SND.append(v)
            fv = np.asarray(p['feat'][i], np.float32)
            k = min(max_cand, c.shape[0], fv.shape[0])
            if k == 0:
                continue
            ct = c[:k, 5] / FPS
            near = nt[np.argmin(np.abs(nt[:, 0][None, :] - ct[:, None]), 1), 1].astype(int)
            F.append(fv[:k]); PL.append(np.clip(near - LO, 0, NP_ - 1))
    return (np.concatenate(F), np.concatenate(PL),
            np.stack(Z), np.stack(SND))


class Head(nn.Module):
    def __init__(self, din=128, hid=128, dout=NP_):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, dout))
        self.register_buffer('mu', torch.zeros(din))
        self.register_buffer('sd', torch.ones(din))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def fit(X, Y, multi, epochs, lr, bs, seed, tag):
    torch.manual_seed(seed)
    m = Head(din=X.shape[1])
    m.mu.copy_(torch.from_numpy(X.mean(0))); m.sd.copy_(
        torch.from_numpy(X.std(0) + 1e-6))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    lossf = nn.BCEWithLogitsLoss() if multi else nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        tot = n = 0.0
        for i in range(0, len(Xt), bs):
            idx = rng.integers(0, len(Xt), bs)
            out = m(Xt[idx])
            loss = lossf(out, Yt[idx] if multi else Yt[idx].long())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        print(f'  {tag} epoch {ep + 1}  loss {tot / max(n, 1):.4f}', flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='/scratch/pmohseni/omr/candf/train_c*.npz')
    ap.add_argument('--test', default='/scratch/pmohseni/omr/candf/room.npz')
    ap.add_argument('--out', default='/scratch/pmohseni/omr/scorer/pitch')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(8)

    print('building training labels...', flush=True)
    F, PL, Z, SND = build_xy(a.train)
    print(f'  score head: {F.shape[0]} candidates, {F.shape[1]}-d')
    print(f'  audio head: {Z.shape[0]} frames, mean {SND.sum(1).mean():.2f} '
          f'pitches sounding', flush=True)
    sh = fit(F, PL, False, a.epochs, 1e-3, 512, a.seed, 'score')
    ah = fit(Z, SND, True, a.epochs, 1e-3, 256, a.seed, 'audio')
    import os
    os.makedirs(a.out, exist_ok=True)
    torch.save({'score': sh.state_dict(), 'audio': ah.state_dict()},
               f'{a.out}/heads_s{a.seed}.pt')

    # ---- the number that matters: separation using PREDICTED pitch ----------
    print('\n=== separation on room, from PREDICTED pitch (no annotations) ===',
          flush=True)
    sh.eval(); ah.eval()
    tr = tw = n = 0.0
    for p in load_feat(a.test):
        if p.get('feat') is None or p.get('z') is None:
            continue
        for i, c in enumerate(p['cand']):
            if c.shape[0] == 0:
                continue
            fv = np.asarray(p['feat'][i], np.float32)
            k = min(64, c.shape[0], fv.shape[0])
            if k == 0:
                continue
            with torch.no_grad():
                ps = torch.softmax(sh(torch.from_numpy(fv[:k])), -1).numpy()
                pa = torch.sigmoid(ah(torch.from_numpy(
                    np.asarray(p['z'][i], np.float32)[None]))).numpy()[0]
            agree = ps @ pa                      # per-candidate scalar
            corr = np.abs(c[:k, 5] - p['t_gt'][i]) <= TH
            if corr.any() and (~corr).any():
                tr += agree[corr].mean(); tw += agree[~corr].mean(); n += 1
    if n:
        print(f'  agreement, CORRECT candidates  {tr / n:.4f}')
        print(f'  agreement, WRONG candidates    {tw / n:.4f}')
        print(f'  ratio                          {tr / max(tw, 1e-9):.2f}x')
        print('  (oracle probe using annotations was 4.70x; this is the real,')
        print('   deployable version, so anything well above 1 is usable)')


if __name__ == '__main__':
    main()


# --------------------------------------------------------------------------
# inference side: turn the two heads into a per-candidate agreement array
# --------------------------------------------------------------------------

def load_heads(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    sh, ah = Head(), Head()
    sh.load_state_dict(d['score']); ah.load_state_dict(d['audio'])
    sh.eval(); ah.eval()
    return sh, ah


def agree(sh, ah, feat, z, k=None):
    """(K,2): [agreement with the sounding pitches, sharpness of this box's own
    pitch prediction]. Nothing symbolic is read -- feat is image, z is audio."""
    fv = np.asarray(feat, np.float32)
    if fv.size == 0:
        return np.zeros((0, 2), np.float32)
    if k is not None:
        fv = fv[:k]
    with torch.no_grad():
        ps = torch.softmax(sh(torch.from_numpy(fv)), -1).numpy()
        pa = torch.sigmoid(ah(torch.from_numpy(
            np.asarray(z, np.float32)[None]))).numpy()[0]
    out = np.zeros((fv.shape[0], 2), np.float32)
    out[:, 0] = ps @ pa
    out[:, 1] = ps.max(1)
    return out


def annotate_pieces(pieces, path):
    """Attach p['pitch'][i] to every frame, once, at load time."""
    sh, ah = load_heads(path)
    n = 0
    for p in pieces:
        if p.get('feat') is None or p.get('z') is None:
            p['pitch'] = None
            continue
        out = []
        for i, c in enumerate(p['cand']):
            fv = p['feat'][i]
            kk = min(c.shape[0], len(fv))
            a = agree(sh, ah, fv, p['z'][i], k=kk)
            if a.shape[0] < c.shape[0]:
                a = np.vstack([a, np.zeros((c.shape[0] - a.shape[0], 2), np.float32)])
            out.append(a)
            n += 1
        p['pitch'] = out
    print(f'[PITCH] agreement attached to {n} frames from {path}', flush=True)
    return pieces
