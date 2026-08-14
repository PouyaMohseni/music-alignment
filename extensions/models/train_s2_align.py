"""S2 training: learned monotonic alignment over the unrolled strip.

THE CONSTRAINT THAT DICTATES THIS LOOP'S SHAPE
-----------------------------------------------
The ColumnEncoder is audio-independent, so encoding the strip once per piece
and reusing it across that piece's chunks is the difference between a feasible
run and an infeasible one -- the strip is ~112 x 19,652 and the encode dominates
everything else.

But "reuse" and "step the optimiser between uses" are incompatible. One
`cols = model.score(strip)` builds ONE autograd graph; the first
`loss.backward()` frees it, and the next chunk raises "backward through the
graph a second time". Detaching `cols` would silence that by cutting the score
tower out of training entirely, which is worse than crashing because it is
silent.

The resolution is to make the unit of optimisation a GROUP of chunks from one
piece: encode the strip once, evaluate `batch` chunks against it, sum into one
loss, one backward, one step. forward_sum_loss is batched precisely so this is
free -- its Python loop over T runs once for the whole group, so B chunks cost
what one chunk costs. The strip encode is then amortised over `batch` chunks
instead of being repeated per chunk.

WHY THE SAMPLER IS CUSTOM
-------------------------
Forward-sum needs ORDERED frames, so the usual shuffled sampler cannot be used.
Chunks are contiguous runs of annotated onsets; groups are shuffled, the frames
inside them are not.
"""
from __future__ import annotations
import argparse, os, random, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.models.monotonic_alignment import (MonotonicAligner, forward_sum_loss,
                                                   causal_viterbi)
from extensions.data.strip_dataset import StripFollowDataset


def chunks_for_piece(ds, pi, chunk):
    """Contiguous frame chunks of EXACTLY `chunk` onsets, ordered by time.

    Exactly, not at-least: chunks are stacked into a (B, T, X) batch, so a
    ragged tail cannot join a group. The dropped remainder is under one chunk
    per piece.
    """
    items = [(i, f) for i, (p, f, _x) in enumerate(ds.items) if p == pi]
    items.sort(key=lambda r: r[1])
    idx = [i for i, _f in items]
    return [idx[i:i + chunk] for i in range(0, len(idx) - chunk + 1, chunk)]


class ChunkGroupSampler(torch.utils.data.Sampler):
    """Yields flat index lists of `batch * chunk` items, all from ONE piece.

    One piece per group is what lets the strip be encoded once per group. The
    group order is shuffled; the order WITHIN each chunk is not, because the
    alignment loss is defined over an ordered sequence.
    """

    def __init__(self, dataset, chunk, batch, shuffle=True):
        self.groups = []
        for pi in range(len(dataset.pieces)):
            chs = chunks_for_piece(dataset, pi, chunk)
            for i in range(0, len(chs) - batch + 1, batch):
                grp = chs[i:i + batch]
                self.groups.append((pi, [j for c in grp for j in c]))
        self.shuffle = shuffle

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.shuffle(order)
        for g in order:
            yield self.groups[g][1]

    def __len__(self):
        return len(self.groups)


def collate_group(batch):
    """Items arrive flat; the caller reshapes to (B, chunk, ...). The strip is
    NOT in here -- see StripFollowDataset(return_strip=False)."""
    _s, mels, tgts, pis = zip(*batch)
    return torch.stack(mels), torch.stack(tgts), pis[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score_dir', default='data/MSMD/cpjku_fmt/score')
    ap.add_argument('--perf_dir',  default='data/MSMD/cpjku_fmt/performance')
    ap.add_argument('--split_dir', default='data/MSMD/cpjku_fmt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ir_bank', default='/scratch/pmohseni/ir_bank')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--batch', type=int, default=4, help='chunks per optimiser step')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--strip_scale', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--anchor_weight', type=float, default=1.0)
    ap.add_argument('--max_step', type=int, default=8)
    ap.add_argument('--jump_cost', type=float, default=8.0)
    # a step here is one ANNOTATED ONSET, not one audio frame: at strip_scale 2
    # and x_stride 8, consecutive noteheads sit ~1-2.5 columns apart, so the
    # forward preference is centred there and kept broad.
    ap.add_argument('--fwd_step', type=float, default=1.5)
    ap.add_argument('--step_sigma', type=float, default=3.0)
    ap.add_argument('--max_pieces', type=int, default=0, help='0 = all (subset probe)')
    ap.add_argument('--max_minutes', type=float, default=0)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)

    def load_split(name):
        d = yaml.safe_load(open(os.path.join(a.split_dir, f'split_{name}.yaml')))
        return d['files'] if isinstance(d, dict) and 'files' in d else d

    tr_p, va_p = load_split('train'), load_split('val')
    if a.max_pieces:
        tr_p, va_p = tr_p[:a.max_pieces], va_p[:max(2, a.max_pieces // 8)]
    print(f'pieces: train={len(tr_p)} val={len(va_p)}  device={dev}', flush=True)

    tr = StripFollowDataset(a.score_dir, a.perf_dir, tr_p, ir_bank=a.ir_bank,
                            augment=True, strip_scale=a.strip_scale, return_strip=False)
    va = StripFollowDataset(a.score_dir, a.perf_dir, va_p, ir_bank=None,
                            augment=False, strip_scale=a.strip_scale, return_strip=False)
    print(f'items: train={len(tr)} val={len(va)}  pieces loaded={len(tr.pieces)}', flush=True)
    if len(tr) == 0:
        raise RuntimeError('no training items -- check score/perf dirs and split names')

    samp = ChunkGroupSampler(tr, a.chunk, a.batch, shuffle=True)
    print(f'groups/epoch={len(samp)}  ({a.batch} chunks x {a.chunk} onsets each)', flush=True)
    if len(samp) == 0:
        raise RuntimeError(f'no groups: need >= {a.batch * a.chunk} onsets in a piece')
    dl = torch.utils.data.DataLoader(
        tr, batch_sampler=samp, num_workers=a.workers, collate_fn=collate_group,
        persistent_workers=a.workers > 0, pin_memory=(dev == 'cuda'))

    model = MonotonicAligner().to(dev)
    print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    stride = model.score.x_stride

    t_start, best, stop = time.time(), 1e9, False
    for ep in range(a.epochs):
        model.train(); tot, n, oom = 0.0, 0, 0
        for mels, tgts, pi in dl:
            B = mels.shape[0] // a.chunk
            t_step = time.time()
            try:
                strip = tr._strip(pi)[None].to(dev)
                cols = model.score(strip)[0]                       # (X, d), ONCE
                win = mels.view(B, a.chunk, *mels.shape[1:]).to(dev, non_blocking=True)
                sim = model.align(win, cols)                       # (B, chunk, X)
                anch = (tgts.to(dev) / stride).round().long().view(B, a.chunk)
                anch = anch.clamp(0, cols.shape[0] - 1)
                loss = forward_sum_loss(sim, anchors=anch, max_step=a.max_step,
                                        jump_cost=a.jump_cost,
                                        anchor_weight=a.anchor_weight,
                                        fwd_step=a.fwd_step, step_sigma=a.step_sigma)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss); n += 1
                # A canary that prints nothing until epoch 0 ends is not a
                # canary: the first epoch's length is exactly what is unknown.
                if n <= 5 or n % 100 == 0:
                    print(f'  [ep{ep} step {n}] {tr.pieces[pi][:26]:26s} '
                          f'X={cols.shape[0]:5d} loss={float(loss):7.4f} '
                          f'{time.time()-t_step:5.2f}s', flush=True)
            except torch.cuda.OutOfMemoryError:
                # the widest strips are ~20k columns; skip rather than die, and
                # report the count so --strip_scale can be raised if it is common
                oom += 1
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                stop = True; print('time budget reached', flush=True); break
        sched.step()

        # validation: causal decode, median column error
        model.eval(); errs = []
        with torch.no_grad():
            for pi in range(len(va.pieces)):
                chs = chunks_for_piece(va, pi, 128)[:4]
                if not chs:
                    continue
                cols = model.score(va._strip(pi)[None].to(dev))[0]
                for ch in chs:
                    mels = torch.stack([va[i][1] for i in ch]).to(dev)
                    tgt = torch.stack([va[i][2] for i in ch])
                    sim = model.align(mels[None], cols)[0]         # (T, X)
                    pred = causal_viterbi(sim, max_step=a.max_step,
                                          jump_cost=a.jump_cost,
                                          fwd_step=a.fwd_step,
                                          step_sigma=a.step_sigma).float().cpu()
                    errs.append((pred - tgt / stride).abs())
        med = float(torch.cat(errs).median()) if errs else float('nan')
        tr_loss = tot / max(1, n)
        print(f'[epoch {ep}] train_loss={tr_loss:.4f}  val_median_col_err={med:.2f}  '
              f'steps={n} oom={oom}  ({(time.time()-t_start)/60:.1f} min)', flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, 'latest.pt'))
        if med == med and med < best:
            best = med
            torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
        if stop:
            break
    print(f'done. best val_median_col_err={best:.2f}', flush=True)


if __name__ == '__main__':
    main()
