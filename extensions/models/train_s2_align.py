"""S2 training: learned monotonic alignment over the unrolled strip."""
from __future__ import annotations
import argparse, os, random, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.models.monotonic_alignment import (MonotonicAligner, forward_sum_loss,
                                                   causal_viterbi)
from extensions.data.strip_dataset import (StripFollowDataset, PieceBatchSampler,
                                           collate_by_piece)


def chunks_for_piece(ds, pi, chunk):
    """Contiguous frame chunks: the forward-sum loss needs ORDERED frames, so
    the usual shuffled sampler cannot be used."""
    items = [(i, f, x) for i, (p, f, x) in enumerate(ds.items) if p == pi]
    items.sort(key=lambda r: r[1])
    return [items[i:i + chunk] for i in range(0, len(items), chunk)
            if len(items[i:i + chunk]) >= 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score_dir', default='data/MSMD/cpjku_fmt/score')
    ap.add_argument('--perf_dir',  default='data/MSMD/cpjku_fmt/performance')
    ap.add_argument('--split_dir', default='data/MSMD/cpjku_fmt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ir_bank', default='/scratch/pmohseni/ir_bank')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
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

    tr = StripFollowDataset(a.score_dir, a.perf_dir, tr_p, ir_bank=a.ir_bank, augment=True)
    va = StripFollowDataset(a.score_dir, a.perf_dir, va_p, ir_bank=None, augment=False)
    print(f'items: train={len(tr)} val={len(va)}  pieces loaded={len(tr.pieces)}', flush=True)
    if len(tr) == 0:
        raise RuntimeError('no training items -- check score/perf dirs and split names')

    model = MonotonicAligner().to(dev)
    print(f'params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    t_start, best = time.time(), 1e9
    for ep in range(a.epochs):
        model.train(); tot, n = 0.0, 0
        order = list(range(len(tr.pieces))); random.shuffle(order)
        for pi in order:
            strip = tr._strip(pi)[None].to(dev)
            cols = model.score(strip)[0]                     # (X, d), once per piece
            for ch in chunks_for_piece(tr, pi, a.chunk):
                mels, tgts = [], []
                for (idx, _f, _x) in ch:
                    _s, mel, t, _p = tr[idx]
                    mels.append(mel); tgts.append(t)
                mel = torch.stack(mels)[None].to(dev)        # (1, T, 78)
                emb = model.audio(mel[0][None])[0]           # (T, d)
                sim = (emb @ cols.T) / model.tau             # (T, X)
                anch = (torch.stack(tgts) / model.score.x_stride).round().long().to(dev)
                loss = forward_sum_loss(sim, anchors=anch)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss); n += 1
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                print('time budget reached', flush=True); break
        sched.step()

        # validation: causal decode, median column error
        model.eval(); errs = []
        with torch.no_grad():
            for pi in range(len(va.pieces)):
                strip = va._strip(pi)[None].to(dev)
                cols = model.score(strip)[0]
                for ch in chunks_for_piece(va, pi, 128)[:4]:
                    mels, tgts = [], []
                    for (idx, _f, _x) in ch:
                        _s, mel, t, _p = va[idx]
                        mels.append(mel); tgts.append(t)
                    emb = model.audio(torch.stack(mels)[None].to(dev))[0]
                    sim = (emb @ cols.T) / model.tau
                    pred = causal_viterbi(sim).float().cpu()
                    true = torch.stack(tgts) / va_stride(model)
                    errs.append((pred - true).abs())
        med = float(torch.cat(errs).median()) if errs else float('nan')
        tr_loss = tot / max(1, n)
        print(f'[epoch {ep}] train_loss={tr_loss:.4f}  val_median_col_err={med:.2f} '
              f'({(time.time()-t_start)/60:.1f} min)', flush=True)
        torch.save(model.state_dict(), os.path.join(a.out, 'latest.pt'))
        if med == med and med < best:
            best = med
            torch.save(model.state_dict(), os.path.join(a.out, 'best.pt'))
        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
            break
    print(f'done. best val_median_col_err={best:.2f}', flush=True)


def va_stride(model):
    return model.score.x_stride


if __name__ == '__main__':
    main()
