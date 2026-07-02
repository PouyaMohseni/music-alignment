"""Precompute DINOv2 column features for MSMD strips.

For each piece, slides an 80×H_strip window with stride 40 across the strip,
resizes each column to 224×224, runs DINOv2-base, and saves the patch tokens.

Output shape per piece: (N_cols, 16, 768)
  N_cols = number of columns (variable per piece)
  16     = patch tokens along the height axis (224/14 = 16)
  768    = DINOv2 hidden dim

Usage:
    python scripts/precompute_dinov2.py \
        --processed data/MSMD/processed \
        --out_dir data/MSMD/dinov2_emb \
        --batch_size 32
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def get_strip_columns(strip_path: str, col_w: int = 80, stride: int = 40,
                      h_target: int = 224) -> np.ndarray:
    """Slide window over strip, return (N_cols, H, W_col, 3) uint8."""
    img = Image.open(strip_path).convert('RGB')
    W, H = img.size
    # Resize to h_target height
    if H != h_target:
        new_w = int(W * h_target / H)
        img = img.resize((new_w, h_target), Image.LANCZOS)
        W = new_w
    arr = np.array(img)   # (H, W, 3)

    n_cols = max(1, (W - col_w) // stride + 1)
    cols = []
    for i in range(n_cols):
        x0 = i * stride
        x1 = min(x0 + col_w, W)
        col = arr[:, x0:x1]   # (H, actual_w, 3)
        # Pad to col_w if near end
        if col.shape[1] < col_w:
            pad = np.zeros((h_target, col_w - col.shape[1], 3), dtype=np.uint8)
            col = np.concatenate([col, pad], axis=1)
        cols.append(col)   # (H, col_w, 3)
    return np.stack(cols, axis=0)   # (N_cols, H, col_w, 3)


@torch.no_grad()
def encode_columns(cols: np.ndarray, processor, model, batch_size: int,
                   device: str) -> np.ndarray:
    """Run DINOv2 on each column (resized to 224×224), return (N_cols, 16, 768).

    The 16 tokens correspond to 224/14 = 16 vertical patch positions
    (mean-pooled over the 16 horizontal positions at each height).
    """
    N, H, W, C = cols.shape
    all_feats = []
    for start in range(0, N, batch_size):
        batch = cols[start:start + batch_size]   # (B, H, W, 3)
        # Convert to PIL list for processor
        pil_imgs = [Image.fromarray(b).resize((224, 224), Image.LANCZOS)
                    for b in batch]
        inputs = processor(images=pil_imgs, return_tensors='pt').to(device)
        out = model(**inputs)
        # patch_tokens: last_hidden_state[:, 1:, :] strips CLS
        # Shape: (B, 256, 768) for 16×16 grid
        patch_tokens = out.last_hidden_state[:, 1:, :]   # (B, 256, 768)
        B, P, D = patch_tokens.shape
        grid_h = grid_w = int(P ** 0.5)   # 16 × 16
        # Reshape to (B, grid_h, grid_w, D), mean-pool over width → (B, 16, D)
        spatial = patch_tokens.reshape(B, grid_h, grid_w, D)
        height_feats = spatial.mean(dim=2)   # (B, 16, 768)
        all_feats.append(height_feats.cpu().float().numpy())

    return np.concatenate(all_feats, axis=0)   # (N_cols, 16, 768)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed', default='data/MSMD/processed')
    p.add_argument('--out_dir',   default='data/MSMD/dinov2_emb')
    p.add_argument('--splits',    nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--col_w',     type=int, default=80)
    p.add_argument('--stride',    type=int, default=40)
    p.add_argument('--device',    default=None)
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading DINOv2...')
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    print(f'  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')

    proc_root = Path(args.processed)
    out_root  = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    splits_data = json.load(open(proc_root / 'splits.json'))
    piece_ids = []
    for s in args.splits:
        piece_ids.extend(splits_data.get(s, []))
    piece_ids = sorted(set(piece_ids))
    print(f'Processing {len(piece_ids)} pieces...')

    n_done = 0
    n_skip = 0
    for pid in piece_ids:
        out_file = out_root / f'{pid}.npy'
        if out_file.exists():
            n_skip += 1
            continue
        strip_path = proc_root / pid / 'strip.png'
        if not strip_path.exists():
            print(f'  SKIP {pid}: strip.png missing')
            continue
        try:
            cols = get_strip_columns(str(strip_path), args.col_w, args.stride)
            feats = encode_columns(cols, processor, model, args.batch_size, device)
            np.save(str(out_file), feats.astype(np.float16))
            n_done += 1
            if n_done % 20 == 0:
                print(f'  [{n_done}/{len(piece_ids) - n_skip}]  {pid}  cols={feats.shape[0]}')
        except Exception as e:
            print(f'  ERROR {pid}: {e}')

    print(f'Done. {n_done} written, {n_skip} skipped.')


if __name__ == '__main__':
    main()
