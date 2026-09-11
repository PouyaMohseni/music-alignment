"""DINOv2-base patch features for every score page, on the detector's own grid.

The scorer's image features come from cyolo_sb's backbone at the candidate's P3
cell, a 52 x 52 grid over the padded page, and they are AUDIO-CONDITIONED:
FiLM runs through that backbone. The swap replaces them with a frozen,
general-purpose visual encoder that never saw audio or music: DINOv2-base at a
728 x 728 input, whose 14-px patches form the SAME 52 x 52 grid, so every
candidate reads a feature at the same spatial resolution as before.

Pages are padded to square exactly as cyolo's load_piece pads them (white
columns split either side), so candidate coordinates in the dumps address
these maps directly. Score pages are static, so one forward pass per page
serves every frame of every performance of it.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

GRID, PATCH = 52, 14
SIDE = GRID * PATCH                       # 728
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def pad_square(sheet):
    """cyolo's load_piece: pad the WIDTH to the height, split around the page."""
    h, w = sheet.shape
    d = abs(h - w)
    return np.pad(sheet, ((0, 0), (d // 2, d - d // 2)), constant_values=255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', nargs='+', required=True, help='piece npz files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    a = ap.parse_args()

    from transformers import AutoModel
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    model = AutoModel.from_pretrained('facebook/dinov2-base').eval()
    os.makedirs(a.out, exist_ok=True)
    for f in sorted(a.npz)[a.shard::a.num_shards]:
        name = os.path.basename(f)[:-4]
        dst = os.path.join(a.out, name + '.npy')
        if os.path.exists(dst):
            continue
        maps = []
        for sheet in np.load(f, allow_pickle=True)['sheets']:
            im = Image.fromarray(pad_square(sheet)).resize((SIDE, SIDE), Image.BOX)
            x = np.asarray(im, np.float32)[..., None].repeat(3, 2) / 255.0
            x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)[None].copy())
            with torch.no_grad():
                h = model(pixel_values=x).last_hidden_state[0, 1:]     # drop CLS
            if h.shape[0] != GRID * GRID:
                raise RuntimeError(f'{name}: {h.shape[0]} patches, expected {GRID * GRID}')
            maps.append(h.reshape(GRID, GRID, -1).numpy().astype(np.float16))
        tmp = dst[:-4] + '.part.npy'
        np.save(tmp, np.stack(maps))
        os.replace(tmp, dst)
        print(f'{name}: {len(maps)} pages', flush=True)


if __name__ == '__main__':
    main()
