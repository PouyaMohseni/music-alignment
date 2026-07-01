"""
Unified training script for v12b / v12c / v12d.
Usage:
    python -m mymodel.v12_mert_align.train_variant --variant v12b ...
    python -m mymodel.v12_mert_align.train_variant --variant v12c ...
    python -m mymodel.v12_mert_align.train_variant --variant v12d ...
"""
import argparse, os, random, time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from mymodel.v12_mert_align.dataset    import MSMDDataset
from mymodel.v12_mert_align.loss       import alignment_loss
from mymodel.v12_mert_align.eval       import eval_piece, aggregate
from mymodel.v12_mert_align.model_variants import V12b, V12c, V12d


def build_model(args):
    if args.variant == 'v12b':
        return V12b(embed_dim=args.embed_dim,
                    lstm_hidden=args.lstm_hidden,
                    lstm_layers=args.lstm_layers)
    if args.variant == 'v12c':
        return V12c(embed_dim=args.embed_dim,
                    lora_rank=args.lora_rank,
                    lstm_hidden=args.lstm_hidden,
                    lstm_layers=args.lstm_layers)
    if args.variant == 'v12d':
        return V12d(embed_dim=args.embed_dim,
                    lora_rank=args.lora_rank,
                    lstm_hidden=args.lstm_hidden,
                    lstm_layers=args.lstm_layers,
                    attn_heads=args.attn_heads)
    raise ValueError(f"Unknown variant: {args.variant}")


def trainable_params(model, variant):
    """Return param groups: lower lr for LoRA, higher for new heads."""
    if variant == 'v12b':
        return [{'params': list(model.lstm.parameters()) +
                           list(model.head.parameters()), 'lr': 3e-4}]
    # v12c, v12d: LoRA gets smaller lr
    lora_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'lora_' in name:
            lora_params.append(p)
        else:
            head_params.append(p)
    return [{'params': lora_params, 'lr': 5e-5},
            {'params': head_params, 'lr': 3e-4}]


def train_epoch(model, dataset, optimizer, device, args):
    model.train()
    model.score_enc.eval()                       # ResNet always frozen
    if args.variant == 'v12b':
        model.audio_enc.eval()                   # MERT frozen in v12b

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    total, bd_sum, n = 0.0, {'infonce': 0.0, 'expected': 0.0}, 0

    for idx in indices:
        p = dataset.pieces[idx]
        if len(p.onset_frames) == 0:
            continue
        wav = p.wav.to(device)
        cols = p.score_cols.to(device)
        of   = torch.from_numpy(p.onset_frames).long().to(device)
        oc   = torch.from_numpy(p.onset_cols).long().to(device)

        optimizer.zero_grad()
        sim = model(wav, cols)
        loss, bd = alignment_loss(sim, of, oc,
                                  w_infonce=args.w_infonce,
                                  w_expected=args.w_expected,
                                  tau=args.tau)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        total += loss.item(); n += 1
        for k in bd: bd_sum[k] += bd[k]

    return total / max(n, 1), {k: v / max(n, 1) for k, v in bd_sum.items()}


def val_epoch(model, dataset, device):
    results = [r for p in dataset.pieces
               if (r := eval_piece(p, model, device))]
    if not results:
        return 0.0, {}
    agg = aggregate(results)
    return agg['pct_0.5s'], agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant',     required=True, choices=['v12b', 'v12c', 'v12d'])
    ap.add_argument('--data_root',   default='data/MSMD/processed')
    ap.add_argument('--out',         default=None)
    ap.add_argument('--epochs',      type=int,   default=30)
    ap.add_argument('--lr',          type=float, default=3e-4)
    ap.add_argument('--w_infonce',   type=float, default=1.0)
    ap.add_argument('--w_expected',  type=float, default=0.5)
    ap.add_argument('--tau',         type=float, default=0.07)
    ap.add_argument('--embed_dim',   type=int,   default=256)
    ap.add_argument('--lstm_hidden', type=int,   default=512)
    ap.add_argument('--lstm_layers', type=int,   default=2)
    ap.add_argument('--lora_rank',   type=int,   default=8)
    ap.add_argument('--attn_heads',  type=int,   default=4)
    ap.add_argument('--patience',    type=int,   default=8)
    ap.add_argument('--device',      default='cuda')
    ap.add_argument('--resume',      default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = f'results/{args.variant}'
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"variant={args.variant}  device={device}  out={args.out}")

    train_set = MSMDDataset('train', args.data_root)
    val_set   = MSMDDataset('val',   args.data_root)

    model = build_model(args).to(device)
    param_groups = trainable_params(model, args.variant)
    n_trainable = sum(p.numel() for g in param_groups for p in g['params'])
    print(f"Trainable params: {n_trainable:,}")

    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_ep, best_val, wait = 0, 0.0, 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_ep  = ckpt['epoch']
        best_val  = ckpt['best_val']
        wait      = ckpt.get('wait', 0)
        print(f"Resumed epoch={start_ep}  best={best_val:.2f}%")

    for epoch in range(start_ep, args.epochs):
        t0 = time.time()
        tr_loss, bd = train_epoch(model, train_set, optimizer, device, args)
        val_pct, val_agg = val_epoch(model, val_set, device)
        scheduler.step()
        lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={tr_loss:.4f}(nce={bd['infonce']:.3f} exp={bd['expected']:.3f})  "
              f"val@0.5s={val_pct:.2f}%  lr={lrs}  {time.time()-t0:.0f}s")

        ckpt = {'state_dict': model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'scheduler':  scheduler.state_dict(),
                'epoch': epoch+1, 'best_val': best_val,
                'wait': wait, 'args': vars(args)}

        if val_pct > best_val:
            best_val, wait = val_pct, 0
            torch.save(ckpt, f'{args.out}/best_model.pt')
            print(f"  -> best {val_pct:.2f}%")
            if val_agg:
                for t in [0.05, 0.1, 0.25, 0.5, 1.0]:
                    print(f"       pct@{t}s={val_agg[f'pct_{t}s']:.2f}%")
        else:
            wait += 1
            print(f"  no improvement ({wait}/{args.patience})")
            if wait >= args.patience:
                print("Early stop.")
                break

        torch.save(ckpt, f'{args.out}/latest.pt')

    print(f"\nBest val pct@0.5s: {best_val:.2f}%")


if __name__ == '__main__':
    main()
