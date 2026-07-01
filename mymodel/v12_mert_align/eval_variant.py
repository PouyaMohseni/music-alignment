"""
Evaluate v12b/v12c/v12d checkpoints on MSMD test split.
Reconstructs the model from the checkpoint's saved args.
"""
import argparse, json, os
import numpy as np
import torch

from mymodel.v12_mert_align.dataset import MSMDDataset, MERT_HZ, col_centers
from mymodel.v12_mert_align.eval import eval_piece, aggregate, THRESHOLDS
from mymodel.v12_mert_align.model_variants import V12b, V12c, V12d


def build_from_ckpt(ckpt):
    saved = ckpt.get('args', {})
    variant = saved.get('variant', 'v12b')
    kw = dict(
        embed_dim   = saved.get('embed_dim',   256),
        lstm_hidden = saved.get('lstm_hidden', 512),
        lstm_layers = saved.get('lstm_layers', 2),
    )
    if variant == 'v12b':
        return V12b(**kw)
    if variant == 'v12c':
        return V12c(lora_rank=saved.get('lora_rank', 8), **kw)
    if variant == 'v12d':
        return V12d(lora_rank=saved.get('lora_rank', 8),
                    attn_heads=saved.get('attn_heads', 4), **kw)
    raise ValueError(f"Unknown variant in checkpoint: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split',      default='test')
    ap.add_argument('--data_root',  default='data/MSMD/processed')
    ap.add_argument('--device',     default='cuda')
    ap.add_argument('--out',        default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_from_ckpt(ckpt).to(device)
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    variant = ckpt.get('args', {}).get('variant', '?')
    print(f"Loaded {variant} checkpoint: {args.checkpoint}")

    dataset = MSMDDataset(args.split, args.data_root)
    results = []
    for i, piece in enumerate(dataset.pieces):
        r = eval_piece(piece, model, device)
        if r:
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(dataset)} pct@0.5s={r['pct_0.5s']:.1f}%")

    agg = aggregate(results)
    print("\n=== Results ===")
    print(f"Variant: {variant}  Pieces: {agg['n_pieces']}  Onsets: {agg['n_onsets']}")
    for thr in THRESHOLDS:
        print(f"  pct within {thr}s : {agg[f'pct_{thr}s']:.1f}%")
    print(f"  mean error  : {agg['mean_error']:.3f}s")
    print(f"  median error: {agg['median_error']:.3f}s")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(agg, open(args.out, 'w'), indent=2)
        print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
