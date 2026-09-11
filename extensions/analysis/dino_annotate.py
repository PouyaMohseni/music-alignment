"""Swap a dump's per-candidate image features for DINOv2 features.

Every key of the source dump is copied unchanged except `feat`/`flens`, which
become a 128-dim PCA projection of the DINOv2-base feature under each
candidate's centre -- for ALL candidates, not only the top FEATK. The scorer is
then retrained on the new dumps with the identical recipe, so the ablation
changes the image encoder and nothing else; 128 dims keeps the scorer's size
identical too.

Candidate x in a dump is UNROLLED (x + add_per_staff[staff]). The page position
is recovered by undoing exactly the unroll compute_batch_stats applied --
staff = nearest staff_coord to y, x = xu - add_per_staff[staff] -- with the
staff tables rebuilt from cyolo's own load_piece output the way load_sequences
builds them. Printed as a check: the share of top-ranked candidates whose
recovered centre lands inside a system box on its page.

  fit       sample candidate features from dumps, fit the PCA
  annotate  write a copy of one dump with DINOv2 features
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
sys.path.insert(0, '/scratch/pmohseni/datasets/cyolo_score_following')

DATA = '/scratch/pmohseni/datasets/cyolo_data/msmd'
DIRS = ('msmd_train', 'msmd_valid', 'msmd_rp')
MAPS = '/scratch/pmohseni/omr/dino_maps'
GRID = 52
CHUNK = 8192


def _data_utils():
    from extensions.hooks.numpy_compat import patch
    patch()
    import cyolo_score_following.utils.data_utils as du
    du.load_wav = lambda *a, **k: np.zeros(1, np.float32)      # geometry only
    return du


class Pages:
    """Staff tables, system boxes and DINOv2 maps, one piece held at a time."""

    def __init__(self):
        self.du = _data_utils()
        self.piece, self.g = None, None

    def get(self, key):
        piece, pg = key.rsplit('_page_', 1)
        if piece != self.piece:
            path = next((f'{DATA}/{d}' for d in DIRS
                         if os.path.exists(f'{DATA}/{d}/{piece}.npz')), None)
            if path is None:
                raise FileNotFoundError(piece)
            padded, _, _, coords, _, systems, _, _, _, _ = self.du.load_piece(path, piece)
            tables = {}
            for page_nr in np.unique(coords[:, -1]):
                pc = coords[coords[:, -1] == page_nr]
                sc = sorted(np.unique(pc[:, 0]))
                max_xes = [0]                      # data_utils.load_sequences
                for c in sc:
                    max_xes.append(max(pc[pc[:, 0] == c, 1]))
                tables[int(page_nr)] = (np.asarray(sc, np.float32),
                                        np.asarray(np.cumsum(max_xes)[:-1], np.float32))
            self.piece = piece
            self.g = dict(tables=tables, H=padded.shape[1], W=padded.shape[2],
                          systems=systems, maps=np.load(f'{MAPS}/{piece}.npy'), cache={})
        g, pg = self.g, int(pg)
        if pg not in g['cache']:
            g['cache'] = {pg: g['maps'][pg].astype(np.float32)}
        return g, pg

    def position(self, key, cand):
        """Page (x, y) of each candidate, undoing the dump's unroll."""
        g, pg = self.get(key)
        sc, aps = g['tables'][pg]
        y = cand[:, 1].astype(np.float32)
        st = np.argmin(np.abs(sc[None, :] - y[:, None]), 1)
        return cand[:, 0].astype(np.float32) - aps[st], y

    def features(self, key, cand):
        g, pg = self.get(key)
        x, y = self.position(key, cand)
        fmap = g['cache'][pg]
        u = np.clip(x / g['W'] * GRID - 0.5, 0, GRID - 1)
        v = np.clip(y / g['H'] * GRID - 0.5, 0, GRID - 1)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        u1, v1 = np.minimum(u0 + 1, GRID - 1), np.minimum(v0 + 1, GRID - 1)
        fu, fv = (u - u0)[:, None], (v - v0)[:, None]
        return ((1 - fv) * ((1 - fu) * fmap[v0, u0] + fu * fmap[v0, u1])
                + fv * ((1 - fu) * fmap[v1, u0] + fu * fmap[v1, u1]))

    def in_system(self, key, cand):
        g, pg = self.get(key)
        x, y = self.position(key, cand)
        boxes = [s for s in g['systems'] if int(s['page_nr']) == pg]
        ok = np.zeros(len(x), bool)
        for s in boxes:
            ok |= ((np.abs(x - s['x']) <= s['w'] / 2 + 20)
                   & (np.abs(y - s['y']) <= s['h'] / 2 + 20))
        return ok


def page_keys(z):
    return sorted({k.split('||')[0] for k in z.files})


def fit(a):
    pages, rng, rows = Pages(), np.random.default_rng(0), []
    for src in a.src:
        z = np.load(src)
        for key in page_keys(z):
            cand, lens = z[f'{key}||cand'], z[f'{key}||lens']
            off = np.concatenate([[0], np.cumsum(lens)])
            for i in range(0, len(lens), a.every):
                if lens[i]:
                    rows.append(pages.features(key, cand[off[i]:off[i + 1]]))
        print(f'{os.path.basename(src)}: {sum(len(r) for r in rows)} rows so far', flush=True)
    X = np.concatenate(rows)
    if len(X) > a.max_rows:
        X = X[rng.choice(len(X), a.max_rows, replace=False)]
    mu = X.mean(0)
    w, V = np.linalg.eigh(np.cov((X - mu).T))
    order = np.argsort(w)[::-1][:a.dim]
    np.savez(a.out, mu=mu.astype(np.float32), comp=V[:, order].astype(np.float32),
             explained=np.float32(w[order].sum() / w.sum()))
    print(f'PCA {X.shape[1]} -> {a.dim} on {len(X)} candidate features: '
          f'{100 * w[order].sum() / w.sum():.1f}% of variance kept -> {a.out}')


def annotate(a):
    p = np.load(a.pca)
    mu, comp = p['mu'], p['comp']
    pages = Pages()
    z = np.load(a.src)
    out = {k: z[k] for k in z.files if not k.endswith(('||feat', '||flens'))}
    top_in, top_n, ncand = 0, 0, 0
    for key in page_keys(z):
        cand, lens = z[f'{key}||cand'], z[f'{key}||lens']
        feats = np.zeros((len(cand), comp.shape[1]), np.float16)
        for s in range(0, len(cand), CHUNK):
            f = pages.features(key, cand[s:s + CHUNK])
            feats[s:s + CHUNK] = ((f - mu) @ comp).astype(np.float16)
        out[f'{key}||feat'] = feats
        out[f'{key}||flens'] = lens.astype(np.int32)
        starts = np.concatenate([[0], np.cumsum(lens)])[:-1][lens > 0]
        if len(starts):
            top_in += int(pages.in_system(key, cand[starts]).sum())
            top_n += len(starts)
        ncand += len(cand)
    os.makedirs(os.path.dirname(a.dst), exist_ok=True)
    tmp = a.dst[:-4] + '.part.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, a.dst)
    print(f'{os.path.basename(a.src)} -> {a.dst}: {len(page_keys(z))} pages, '
          f'{ncand} candidates; top-ranked candidate inside a system box: '
          f'{100.0 * top_in / max(top_n, 1):.1f}% of {top_n} frames')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    f = sp.add_parser('fit')
    f.add_argument('--src', nargs='+', required=True)
    f.add_argument('--out', required=True)
    f.add_argument('--dim', type=int, default=128)
    f.add_argument('--every', type=int, default=25, help='frame stride when sampling')
    f.add_argument('--max_rows', type=int, default=200000)
    n = sp.add_parser('annotate')
    n.add_argument('--pca', required=True)
    n.add_argument('--src', required=True)
    n.add_argument('--dst', required=True)
    a = ap.parse_args()
    fit(a) if a.cmd == 'fit' else annotate(a)


if __name__ == '__main__':
    main()
