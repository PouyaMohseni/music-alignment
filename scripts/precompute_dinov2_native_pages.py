"""Precompute DINOv2 global page embeddings for native MSMD score pages
(third_party/cpjku_unet/data/msmd/{msmd_train_full-equivalent,msmd_valid,msmd_test}/score/<pid>_page_N.npz).

Unlike MERT's audio embeddings (one per (piece, tempo_factor) since audio
changes with tempo), the score image does NOT change with tempo -- one
embedding per page is enough, no tempo loop needed. This also means it
does NOT need FluidSynth/audio rendering at all, just a resize + a single
DINOv2 forward pass per page -- much cheaper than the MERT precompute.

Saves BOTH the CLS token (768,) -- for a low-risk global-context injection
at the network's bottleneck, everything else in the architecture untouched
-- AND the full 16x16 patch-token grid (256, 768) -- for the higher-risk
experiment that replaces the entire from-scratch multi-scale visual encoder
with a frozen-DINOv2 + trainable adapter "neck" pyramid. One precompute
pass serves both.

Runs in the MAIN project venv (has torch/transformers); venv_cpjku310 (used
for actual training) has torch but not transformers, so live DINOv2 can't
run there -- confirmed directly before writing this script, same
constraint MERT already had.

    python scripts/precompute_dinov2_native_pages.py \
        --score_dirs /scratch/pmohseni/msmd_train_full/score third_party/cpjku_unet/data/msmd/msmd_valid/score third_party/cpjku_unet/data/msmd/msmd_test/score \
        --out_dir /scratch/pmohseni/dinov2_emb_native
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@torch.no_grad()
def encode_page(sheet: np.ndarray, processor, model, device: str) -> tuple[np.ndarray, np.ndarray]:
    """sheet: (H, W) uint8, native page image.
    Returns (cls: (768,), patch_grid: (16, 16, 768))."""
    img = Image.fromarray(sheet).convert('RGB').resize((224, 224), Image.LANCZOS)
    inputs = processor(images=[img], return_tensors='pt').to(device)
    out = model(**inputs)
    cls = out.last_hidden_state[:, 0, :]           # (1, 768)
    patch_tokens = out.last_hidden_state[:, 1:, :]  # (1, 256, 768)
    P, D = patch_tokens.shape[1], patch_tokens.shape[2]
    grid = int(P ** 0.5)   # 16
    patch_grid = patch_tokens.reshape(1, grid, grid, D)[0]   # (16, 16, 768)
    return cls[0].cpu().float().numpy(), patch_grid.cpu().float().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--score_dirs', nargs='+', required=True,
                   help='one or more score/ directories containing <pid>_page_N.npz files')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', default=None)
    a = p.parse_args()

    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    print('Loading DINOv2-base...', flush=True)
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    model = AutoModel.from_pretrained('facebook/dinov2-base').to(device).eval()
    print(f'  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M', flush=True)

    out_dir = Path(a.out_dir)
    cls_dir = out_dir / 'cls'
    grid_dir = out_dir / 'grid'
    cls_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = []
    for d in a.score_dirs:
        npz_paths.extend(sorted(Path(d).glob('*.npz')))
    print(f'{len(npz_paths)} native pages to process', flush=True)

    done = skip = fail = 0
    for i, npz_path in enumerate(npz_paths):
        pid = npz_path.stem   # e.g. AndreJ__O34__andre-sonatine_page_0
        cls_path = cls_dir / f'{pid}.npy'
        grid_path = grid_dir / f'{pid}.npy'
        if cls_path.exists() and grid_path.exists():
            skip += 1
            continue
        try:
            npz = np.load(npz_path, allow_pickle=True)
            sheet = npz['sheet']
            cls, grid = encode_page(sheet, processor, model, device)
            np.save(cls_path, cls.astype(np.float16))
            np.save(grid_path, grid.astype(np.float16))
            done += 1
        except Exception as e:
            print(f'  FAIL {pid}: {e}', flush=True)
            fail += 1
        if (i + 1) % 200 == 0:
            print(f'  [{i+1}/{len(npz_paths)}] done={done} skip={skip} fail={fail}', flush=True)

    print(f'Done. done={done} skip={skip} fail={fail} total={len(npz_paths)}', flush=True)


if __name__ == '__main__':
    main()
