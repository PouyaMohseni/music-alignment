"""Precompute DINOv2 TILED features for native MSMD score pages -- the
resolution-preserving version for the full-encoder-replacement experiment.

The whole-page-squashed-to-224x224 approach (precompute_dinov2_native_pages.py)
resizes an entire ~1181x835 page down to 224x224 before DINOv2 ever sees it,
then further divides that into a 16x16 patch grid -- each patch would span
a large fraction of the page, blurring together many staff lines/noteheads.
Sheet music needs much finer effective resolution than that to be useful
for pixel-precise localization.

This instead tiles each page at (close to) NATIVE resolution: divide the
page into a grid of TILE_SIZE x TILE_SIZE crops (no page-level downscaling
first), resize only each individual tile to 224x224 for DINOv2 (a much
smaller upscale/downscale factor than squashing the whole page), and take
each tile's CLS token as that tile's feature. Output is a genuine
(n_rows, n_cols, 768) 2D grid whose n_rows/n_cols scale with the actual
page size (varies per piece, same variable-size handling the existing
from-scratch conv encoder already does -- no fixed-size assumption here
either, consistent with that).

    python scripts/precompute_dinov2_tiled_native.py \
        --score_dirs /scratch/pmohseni/msmd_train_full/score third_party/cpjku_unet/data/msmd/msmd_valid/score third_party/cpjku_unet/data/msmd/msmd_test/score \
        --out_dir /scratch/pmohseni/dinov2_emb_tiled_native \
        --tile_size 224 --stride 224
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def get_page_tiles(sheet: np.ndarray, tile_size: int, stride: int) -> tuple[np.ndarray, int, int]:
    """sheet: (H, W) uint8. Returns (tiles: (n_rows*n_cols, tile_size, tile_size, 3) uint8,
    n_rows, n_cols) -- tiles pad with zeros (background) past the page edge
    rather than resizing the whole page, to preserve native pixel density."""
    H, W = sheet.shape
    # ceil-based tiling covering the whole page (1 tile if the page is
    # already smaller than tile_size in that dimension)
    n_rows = max(1, -(-max(H - tile_size, 0) // stride) + 1)
    n_cols = max(1, -(-max(W - tile_size, 0) // stride) + 1)

    rgb = np.stack([sheet, sheet, sheet], axis=-1)   # (H, W, 3), grayscale -> RGB

    tiles = []
    for r in range(n_rows):
        y0 = min(r * stride, max(0, H - tile_size)) if H > tile_size else 0
        y1 = y0 + tile_size
        for c in range(n_cols):
            x0 = min(c * stride, max(0, W - tile_size)) if W > tile_size else 0
            x1 = x0 + tile_size
            tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            src = rgb[y0:min(y1, H), x0:min(x1, W)]
            tile[:src.shape[0], :src.shape[1]] = src
            tiles.append(tile)

    return np.stack(tiles, axis=0), n_rows, n_cols


@torch.no_grad()
def encode_tiles(tiles: np.ndarray, processor, model, batch_size: int, device: str) -> np.ndarray:
    """tiles: (N, tile_size, tile_size, 3) uint8. Returns (N, 768) CLS embeddings."""
    all_cls = []
    for start in range(0, tiles.shape[0], batch_size):
        batch = tiles[start:start + batch_size]
        pil_imgs = [Image.fromarray(t).resize((224, 224), Image.LANCZOS) for t in batch]
        inputs = processor(images=pil_imgs, return_tensors='pt').to(device)
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0, :]   # (B, 768)
        all_cls.append(cls.cpu().float().numpy())
    return np.concatenate(all_cls, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--score_dirs', nargs='+', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--tile_size', type=int, default=224)
    p.add_argument('--stride', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device', default=None)
    a = p.parse_args()

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    print('Loading DINOv2-base...', flush=True)
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    print(f'  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M', flush=True)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = []
    for d in a.score_dirs:
        npz_paths.extend(sorted(Path(d).glob('*.npz')))
    print(f'{len(npz_paths)} native pages to process (tile_size={a.tile_size}, stride={a.stride})', flush=True)

    done = skip = fail = 0
    for i, npz_path in enumerate(npz_paths):
        pid = npz_path.stem
        out_path = out_dir / f'{pid}.npy'
        meta_path = out_dir / f'{pid}_meta.npy'
        if out_path.exists() and meta_path.exists():
            skip += 1
            continue
        try:
            npz = np.load(npz_path, allow_pickle=True)
            sheet = npz['sheet']
            tiles, n_rows, n_cols = get_page_tiles(sheet, a.tile_size, a.stride)
            cls = encode_tiles(tiles, processor, model, a.batch_size, device)   # (n_rows*n_cols, 768)
            grid = cls.reshape(n_rows, n_cols, 768)
            np.save(out_path, grid.astype(np.float16))
            np.save(meta_path, np.array([n_rows, n_cols, sheet.shape[0], sheet.shape[1]]))
            done += 1
        except Exception as e:
            print(f'  FAIL {pid}: {e}', flush=True)
            fail += 1
        if (i + 1) % 100 == 0:
            print(f'  [{i+1}/{len(npz_paths)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Done. done={done} skip={skip} fail={fail} total={len(npz_paths)}', flush=True)


if __name__ == '__main__':
    main()
