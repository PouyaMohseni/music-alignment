"""DINOv2 in place of CYOLO's visual stem, the image-side twin of the MERT swap.

WHY THIS EXPERIMENT EXISTS
--------------------------
We replace the perception model's AUDIO encoder two ways, with MERT and with
CODA's Mamba, and both leave the real-audio result far behind what re-deciding
over the same candidates achieves. The image side had no equivalent: what the
paper called an image ablation swapped the features the DECISION model reads,
not the encoder inside the perception model. That is a different test, and
without this one the claim that perception is not the bottleneck rests on audio
alone.

THE SWAP IS A DROP-IN, BY ARITHMETIC
------------------------------------
cyolo_sb.yaml sends a single 52 x 52 feature map to the Detect layer for all
three classes, reached by Focus and three strided convolutions from a 416 x 416
page. DINOv2-base at 728 x 728 has 14-px patches, which is the same 52 x 52
grid. So layers 0-4 (stem through P3, 64 channels at 52 x 52) can be replaced
by a 1x1 projection of the DINOv2 map, and layers 5-8 (the FiLM downscales),
the FPN, the anchors and the heads all run untouched. Everything conditioned on
audio stays exactly where it was, so a change is attributable to the image
representation and nothing else.

Applied AFTER Model.__init__ for two reasons. initialize_weights orthogonalises
every Linear and Conv2d, and more importantly __init__ computes the Detect
strides by running a dummy 256 x 256 image through the network; with the stem
replaced that dummy has the wrong channel count, and the strides it would infer
would be wrong even if it ran. Patching afterwards keeps the original stride 8,
which stays correct: detections live on a 52 x 52 grid in both architectures.

AUGMENTATION
------------
cyolo shifts the page by a random pixel offset and shifts the labels with it. A
precomputed patch map can only be rolled in units of 8 scaled pixels, and the
note anchors are 11 px wide, so a sub-patch roll would desynchronise features
from labels by most of a notehead. This arm therefore trains without image
shifts, and its control must too -- see train_dino_enc.sh, which launches both.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

MAPS = os.environ.get('DINO_MAPS', '/scratch/pmohseni/omr/dino_maps')
GRID, DIM = 52, 768


@lru_cache(maxsize=24)
def _page_maps(piece: str) -> np.ndarray:
    """(n_pages, 52, 52, 768) float16, one file per piece, cached by piece."""
    return np.load(os.path.join(MAPS, piece + '.npy'), mmap_mode='r')


class DINOv2Stem(nn.Module):
    """1x1 projection standing in for Focus and the three strided convs.

    Output must match what layer 4 produced, 64 channels at 52 x 52, because
    the FPN concatenates it and layer 5 consumes it.
    """

    def __init__(self, out_ch=64, groupnorm=True):
        super().__init__()
        self.proj = nn.Conv2d(DIM, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, out_ch) if groupnorm else nn.BatchNorm2d(out_ch)
        self.act = nn.ELU(False)

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


def patch_dataset():
    """Hand the network the map instead of the page, leaving labels alone.

    __getitem__ normalises the targets against the 416-px image before we see
    the sample, so replacing sample['score'] afterwards changes the network's
    input and nothing else.
    """
    import cyolo_score_following.dataset as ds

    prev = ds.SequenceDataset.__getitem__

    def __getitem__(self, item):
        sample = prev(self, item)
        piece, page = sample['file_name'].rsplit('_page_', 1)
        m = _page_maps(piece)[int(page)]                 # (52, 52, 768)
        sample['score'] = np.ascontiguousarray(
            np.transpose(np.asarray(m, np.float32), (2, 0, 1)))
        return sample

    ds.SequenceDataset.__getitem__ = __getitem__
    ds._dino_patched = True
    print(f'[DINO] dataset serving {DIM}-d maps from {MAPS}', flush=True)


def patch_model(out_ch=64):
    from cyolo_score_following.models.yolo import Model

    if getattr(Model, '_dino_patched', False):
        return
    prev_init = Model.__init__

    def __init__(self, *a, **kw):
        prev_init(self, *a, **kw)
        gn = bool(self.yaml.get('groupnorm', True))
        stem = DINOv2Stem(out_ch=out_ch, groupnorm=gn)
        # the forward loop reads .f (input source) and .i (index) off every
        # module, so the replacements inherit them from what they replace
        old0 = self.model[0]
        stem.f, stem.i = old0.f, old0.i
        n_old = sum(p.numel() for p in self.model[:5].parameters())
        self.model[0] = stem
        for k in range(1, 5):
            ident = nn.Identity()
            ident.f, ident.i = self.model[k].f, self.model[k].i
            self.model[k] = ident
        n_new = sum(p.numel() for p in self.model[:5].parameters())
        print(f'[DINO] visual stem: layers 0-4 ({n_old} params) -> 1x1 '
              f'projection {DIM}->{out_ch} ({n_new} params); Detect stride '
              f'{getattr(self.model[-1], "stride", None)} kept', flush=True)

    Model.__init__ = __init__
    Model._dino_patched = True


def patch_noshift():
    """Turn off the page shift for BOTH arms, keeping tempo augmentation.

    train.py's single --augment flag drives two unrelated things: the dataset's
    image shift plus random audio padding, and the spectrogram tempo stretch
    passed to iterate_dataset. Dropping the flag would remove tempo
    augmentation as well and change far more than intended, so the flag stays
    on and the dataset is built with augment=False instead.
    """
    import cyolo_score_following.dataset as ds

    prev = ds.load_dataset

    def load_dataset(*a, **kw):
        kw['augment'] = False
        return prev(*a, **kw)

    ds.load_dataset = load_dataset
    ds._noshift_patched = True
    print('[DINO] page shift off for this arm (tempo augmentation kept)',
          flush=True)


def maybe_patch_dino() -> bool:
    if os.environ.get('DINO_NOSHIFT', '0') != '0':
        patch_noshift()
    if os.environ.get('DINO_ENC', '0') == '0':
        return False
    patch_dataset()
    patch_model(out_ch=int(os.environ.get('DINO_OUT_CH', '64')))
    return True
