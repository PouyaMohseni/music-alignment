"""
Train v12 MERT alignment model.

- One piece per step (variable length; no padding needed)
- MERT and ResNet18 encoders stay frozen; only AlignmentHead is trained
- Loss: InfoNCE + expected-position
- Eval: pct_within_0.5s on val set after each epoch
- Saves best checkpoint on val pct@0.5s
"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from mymodel.v12_mert_align.dataset import MSMDDataset, MERT_HZ
from mymodel.v12_mert_align.model   import MERTAlignModel
from mymodel.v12_mert_align.loss    import alignment_loss
from mymodel.v12_mert_align.eval    import eval_piece, aggregate


def train_epoch(model, dataset, optimizer, device, args):
    model.train()
    model.audio_enc.eval()   # MERT always in eval (frozen BN/LN)
    model.score_enc.eval()   # ResNet always in eval

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    total_loss = 0.0
    breakdown  = {'infonce': 0.0, 'expected': 0.0}
    n = 0

    for idx in indices:
        piece = dataset.pieces[idx]
        if len(piece.onset_frames) == 0:
            continue

        wav  = piece.wav.to(device)
        cols = piece.score_cols.to(device)
        of   = torch.from_numpy(piece.onset_frames).long().to(device)
        oc   = torch.from_numpy(piece.onset_cols).long().to(device)

        optimizer.zero_grad()
        sim = model(wav, cols)
        loss, bd = alignment_loss(sim, of, oc,
                                  w_infonce=args.w_infonce,
                                  w_expected=args.w_expected,
                                  tau=args.tau)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k in bd:
            breakdown[k] += bd[k]
        n += 1

    return total_loss / max(n, 1), {k: v / max(n, 1) for k, v in breakdown.items()}


def val_epoch(model, dataset, device):
    results = []
    for piece in dataset.pieces:
        r = eval_piece(piece, model, device)
        if r:
            results.append(r)
    if not results:
        return 0.0, {}
    agg = aggregate(results)
    return agg['pct_0.5s'], agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',   default='data/MSMD/processed')
    ap.add_argument('--out',         default='results/v12_mert_align')
    ap.add_argument('--epochs',      type=int,   default=30)
    ap.add_argument('--lr',          type=float, default=3e-4)
    ap.add_argument('--w_infonce',   type=float, default=1.0)
    ap.add_argument('--w_expected',  type=float, default=0.5)
    ap.add_argument('--tau',         type=float, default=0.07)
    ap.add_argument('--embed_dim',   type=int,   default=256)
    ap.add_argument('--patience',    type=int,   default=10)
    ap.add_argument('--device',      default='cuda')
    ap.add_argument('--resume',      default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  out={args.out}")

    train_set = MSMDDataset('train', args.data_root)
    val_set   = MSMDDataset('val',   args.data_root)

    model = MERTAlignModel(embed_dim=args.embed_dim).to(device)
    # Only train the projection head
    trainable = [p for p in model.head.parameters()]
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    start_epoch = 0
    best_val    = 0.0
    wait        = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_val    = ckpt['best_val']
        wait        = ckpt.get('wait', 0)
        print(f"Resumed from epoch {start_epoch}, best_val={best_val:.2f}%")

    trainable_params = sum(p.numel() for p in model.head.parameters())
    print(f"Trainable params (head only): {trainable_params:,}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, bd = train_epoch(model, train_set, optimizer, device, args)
        val_pct, val_agg = val_epoch(model, val_set, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={train_loss:.4f} "
              f"(nce={bd['infonce']:.4f} exp={bd['expected']:.4f})  "
              f"val_pct@0.5s={val_pct:.2f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"{elapsed:.0f}s")

        improved = val_pct > best_val
        if improved:
            best_val = val_pct
            wait = 0
            ckpt_path = os.path.join(args.out, 'best_model.pt')
            torch.save({'state_dict': model.state_dict(),
                        'optimizer':  optimizer.state_dict(),
                        'scheduler':  scheduler.state_dict(),
                        'epoch':      epoch + 1,
                        'best_val':   best_val,
                        'wait':       wait,
                        'args':       vars(args)}, ckpt_path)
            print(f"  -> new best ({val_pct:.2f}%), saved {ckpt_path}")
            if val_agg:
                for thr in [0.05, 0.1, 0.25, 0.5, 1.0]:
                    print(f"       pct@{thr}s = {val_agg[f'pct_{thr}s']:.2f}%")
        else:
            wait += 1
            print(f"  no improvement ({wait}/{args.patience})")
            if wait >= args.patience:
                print("Early stopping.")
                break

        # Always save latest
        torch.save({'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'scheduler':  scheduler.state_dict(),
                    'epoch':      epoch + 1,
                    'best_val':   best_val,
                    'wait':       wait,
                    'args':       vars(args)},
                   os.path.join(args.out, 'latest.pt'))

    print(f"\nBest val pct@0.5s: {best_val:.2f}%")


if __name__ == '__main__':
    main()
