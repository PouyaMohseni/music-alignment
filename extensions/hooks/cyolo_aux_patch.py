"""C1 -- attach posteriorgram distillation to CYOLO's training, nothing else.

Three patches, no third-party file edited:

  1. `SequenceDataset.__getitem__` attaches `post_target` -- the posteriorgram
     row at the sample's own frame.
  2. `CustomBatch` stacks it (and `pin_memory` moves it).
  3. `compute_loss` gains an auxiliary term computed from the conditioning
     vector z.

WHY THE BASE IS THE RELEASED CHECKPOINT, NOT OUR REPRODUCTION
-------------------------------------------------------------
`trained_models/cyolo_sb/best_model.pt` IS the 79.9 model, and we verified that
in our own harness (63.0 at 0.1 s / 79.9 at 0.5 s, matching the published row
exactly, with --only_onsets). Fine-tuning from it means the delta this run
produces is attributable to the auxiliary loss alone, with zero reproduction
variance -- and it costs no GPU to re-derive a number that already exists.

INFERENCE IS UNCHANGED
----------------------
The head is only ever touched inside compute_loss. `Model.forward` is not
patched, so eval.py runs the stock network with stock weights and the stock
decode. The head's parameters live in the checkpoint (they are registered
submodules) but no eval path reads them.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

POST_DIM = 176


def _load_bank(root: str) -> dict:
    files = sorted(Path(root).glob('*.npy'))
    if not files:
        raise RuntimeError(f'posteriorgram bank {root!r} contains no .npy files')
    return {f.stem: str(f) for f in files}


def patch_cyolo_aux(bank_roots, weight: float = 1.0, zdim: int = 128, strict: bool = True):
    """bank_roots: {dataset_dir: posteriorgram_dir}. Call BEFORE load_dataset."""
    import cyolo_score_following.dataset as ds_mod
    import cyolo_score_following.utils.loss as loss_mod
    from extensions.heads.posteriorgram_distill import (PosteriorgramDistillHead,
                                                        distill_loss)

    banks = {k: _load_bank(v) for k, v in bank_roots.items()}
    print('[C1] posteriorgram banks: ' +
          ', '.join(f'{k}->{len(v)} npy' for k, v in banks.items()), flush=True)

    cache = {}

    def _post_for(piece_name):
        if piece_name in cache:
            return cache[piece_name]
        for b in banks.values():
            if piece_name in b:
                arr = np.load(b[piece_name]).astype(np.float32)
                if arr.shape[1] != POST_DIM:
                    raise RuntimeError(f'{piece_name}: expected {POST_DIM} dims, got {arr.shape}')
                if len(cache) < 512:
                    cache[piece_name] = arr
                return arr
        return None

    # ---- 1. dataset attaches the target ------------------------------------
    _orig_getitem = ds_mod.SequenceDataset.__getitem__
    _missing = set()

    def __getitem__(self, item):
        sample = _orig_getitem(self, item)
        seq = self.sequences[item]
        piece = self.piece_names[seq['piece_id']]
        post = _post_for(piece)
        if post is None:
            if piece not in _missing:
                _missing.add(piece)
                msg = f'[C1] no posteriorgram for {piece!r}'
                if strict:
                    raise RuntimeError(msg + ' -- refusing to train with a partial '
                                             'target bank, which would silently apply '
                                             'the aux loss to only some pieces')
                print('WARNING ' + msg, flush=True)
            sample['post_target'] = np.zeros(POST_DIM, dtype=np.float32)
            sample['post_valid'] = False
        else:
            # the sample's own frame; clamp because the dataset builds ~2 more
            # sequences than the bank has frames (the tail windows), exactly as
            # native CYOLO does when the signal runs out
            f = min(int(seq['frame']), post.shape[0] - 1)
            sample['post_target'] = post[f]
            sample['post_valid'] = True
        return sample

    ds_mod.SequenceDataset.__getitem__ = __getitem__
    ds_mod.SequenceDataset._c1_patched = True

    # ---- 2. collate carries it ---------------------------------------------
    _orig_batch_init = ds_mod.CustomBatch.__init__
    _orig_pin = ds_mod.CustomBatch.pin_memory

    def batch_init(self, batch):
        _orig_batch_init(self, batch)
        self.post_target = torch.as_tensor(
            np.stack([x['post_target'] for x in batch]), dtype=torch.float32)
        self.post_valid = torch.as_tensor(
            np.asarray([x['post_valid'] for x in batch]), dtype=torch.bool)

    def pin_memory(self):
        _orig_pin(self)
        self.post_target = self.post_target.pin_memory()
        return self

    ds_mod.CustomBatch.__init__ = batch_init
    ds_mod.CustomBatch.pin_memory = pin_memory

    # ---- 3. capture z, stash the target, add the aux term ------------------
    # criterion is called as criterion(pred, targets, network) -- no z, no
    # batch. Rather than change that signature (and every call site), z is
    # captured by a forward hook on z_enc and the batch target is stashed on the
    # network by a patched iterate_dataset. compute_loss then reads both off the
    # model.
    _orig_compute_loss = loss_mod.compute_loss

    def _attach_z_hook(network):
        net = network.module if hasattr(network, 'module') else network
        if getattr(net, '_c1_z_hooked', False):
            return net
        cn = net.conditioning_network

        def _hook(module, inp, out):
            net._c1_z = out

        cn.z_enc.register_forward_hook(_hook)
        net._c1_z_hooked = True
        return net

    def compute_loss(p, targets, model):
        loss_dict = _orig_compute_loss(p, targets, model)
        net = model.module if hasattr(model, 'module') else model
        z = getattr(net, '_c1_z', None)
        tgt = getattr(net, '_c1_post_target', None)
        if z is None or tgt is None:
            return loss_dict
        head = getattr(net, '_c1_distill_head', None)
        if head is None:
            head = PosteriorgramDistillHead(zdim=z.shape[-1], out_dim=POST_DIM).to(z.device)
            net.add_module('_c1_distill_head', head)
            print(f'[C1] distill head created ({z.shape[-1]} -> {POST_DIM})', flush=True)
        # z is (N, zdim) with N = batch size, one conditioning vector per sample
        if z.shape[0] != tgt.shape[0]:
            raise RuntimeError(f'[C1] z has {z.shape[0]} rows but the target has '
                               f'{tgt.shape[0]} -- they must be one per sample')
        aux, _ = distill_loss(head(z), tgt.to(z.device),
                              getattr(net, '_c1_post_valid', None))
        loss_dict['loss'] = loss_dict['loss'] + weight * aux
        loss_dict['aux_distill'] = aux.detach()
        return loss_dict

    loss_mod.compute_loss = compute_loss
    loss_mod._c1_patched = True

    # ---- 4. training loop: stash the target, register the head -------------
    _orig_iterate = ds_mod.iterate_dataset

    def iterate_dataset(network, dataloader, criterion, optimizer=None, **kw):
        net = _attach_z_hook(network)
        _orig_getattr_batch = None

        # wrap the dataloader so each batch's target lands on the network just
        # before the forward that consumes it
        class _Wrapped:
            def __init__(self, dl):
                self.dl = dl

            def __len__(self):
                return len(self.dl)

            def __iter__(self):
                for data in self.dl:
                    net._c1_post_target = data.post_target
                    net._c1_post_valid = data.post_valid
                    # the head is built on the first compute_loss call, which is
                    # AFTER the optimizer was constructed -- register it then
                    if optimizer is not None and getattr(net, '_c1_distill_head', None) \
                            is not None and not getattr(net, '_c1_registered', False):
                        optimizer.add_param_group(
                            {'params': net._c1_distill_head.parameters()})
                        net._c1_registered = True
                        print('[C1] distill head registered with the optimizer', flush=True)
                    yield data

        return _orig_iterate(network, _Wrapped(dataloader), criterion,
                             optimizer=optimizer, **kw)

    ds_mod.iterate_dataset = iterate_dataset
    ds_mod._c1_iterate_patched = True

    print(f'[C1] posteriorgram distillation ACTIVE (weight={weight})', flush=True)
    return {'ok': True}
