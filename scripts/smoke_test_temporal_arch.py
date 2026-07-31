"""Smoke test for the N1/N2/N3 temporal-architecture experiments.

Verifies the properties the whole design rests on, BEFORE spending GPU time:

 1. Each variant builds and forward-passes with correct output shapes.
 2. The returned `hidden` matches the (network.rnn_layers, bs, network.rnn_size)
    contract that CPJKU's iterate_dataset allocates and slices -- if this is
    wrong, training crashes on the second chunk or silently corrupts state.
 3. N2 and N3 are EXACTLY equal to stock CB_TA at initialisation (zero-init
    gate). This is the load-bearing claim: it is what makes warm-starting
    from B1a's converged checkpoint meaningful, and what guarantees these
    runs cannot start below the 89.2% baseline the way every FiLM
    replacement did.
 4. State survives chunk boundaries, a per-slot reset (hidden[i][:, idx] = 0,
    how iterate_dataset starts a new piece) and a slot drop (the dim-1
    concatenate it does when a piece finishes).
 5. No NaNs -- the masked-memory attention in N1 is the classic place to get
    them (softmax over an all -inf row).
 6. Gradients actually reach every new module (a zero-init gate that also
    kills its own gradient would never train).

Run: python -m scripts.smoke_test_temporal_arch
"""
from __future__ import annotations
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'cpjku_unet'))

from audio_conditioned_unet import network as cpjku_network   # noqa: E402
from extensions.hooks.temporal_arch_patch import (   # noqa: E402
    patch_long_context_temporal,
    patch_gated_memory_retrieval,
    patch_gated_belief_propagation,
)

CONFIG = dict(n_encoder_layers=4, n_filters_start=8, use_lstm=True, rnn_size=64,
              rnn_layer=1, spec_enc=64, audio_encoder='CBEncoder',
              film_layers=[2, 3, 4, 5, 6, 7, 8])
SEQ_LEN, BS, H, W = 4, 2, 96, 128
OPTS = dict(n_layers=2, n_heads=4, n_fine=8, n_comp=12, pool=4, n_mem=12,
            belief_h=8, belief_w=16)

_StockUNet = cpjku_network.ConditionalUNet


def _inputs(seq_len=SEQ_LEN, bs=BS):
    torch.manual_seed(1234)
    score = torch.rand(seq_len, bs, 1, H, W)
    perf = torch.rand(seq_len, bs, 1, 78, 40)
    return score, perf


def _zeros_hidden(net, bs=BS):
    """Exactly how iterate_dataset allocates it."""
    return (torch.zeros(net.rnn_layers, bs, net.rnn_size),
            torch.zeros(net.rnn_layers, bs, net.rnn_size))


def _build(kind):
    """Build via the PUBLIC patch entry point, not _build_network directly.

    The patch function is also what wraps network.initialize_weights to
    honour `_zero_init` tags; calling the builder directly skips that, so
    ConditionalUNet.__init__'s trailing self.apply(initialize_weights)
    orthogonal-inits the branch's deliberately zeroed final layer and the
    identity check fails for a reason that does not exist in training. Going
    through the real entry point keeps the test faithful to production."""
    patch = {
        'long_context': patch_long_context_temporal,
        'memory_retrieval': patch_gated_memory_retrieval,
        'belief': patch_gated_belief_propagation,
    }[kind]
    patch(**OPTS)          # installs cpjku_network.ConditionalUNet + the init guard
    torch.manual_seed(0)
    return cpjku_network.ConditionalUNet(CONFIG)


def check_shapes_and_state(kind):
    net = _build(kind).eval()
    score, perf = _inputs()
    hidden = _zeros_hidden(net)
    with torch.no_grad():
        out = net(score=score, perf=perf, hidden=hidden)
    seg, new_hidden = out['segmentation'], out['hidden']

    assert seg.shape == (SEQ_LEN * BS, 1, H, W), f'{kind}: seg shape {seg.shape}'
    assert torch.isfinite(seg).all(), f'{kind}: non-finite segmentation'
    for i, t in enumerate(new_hidden):
        assert t.shape == (net.rnn_layers, BS, net.rnn_size), \
            f'{kind}: hidden[{i}] {tuple(t.shape)} != {(net.rnn_layers, BS, net.rnn_size)}'
        assert torch.isfinite(t).all(), f'{kind}: non-finite hidden[{i}]'
    print(f'  [{kind}] shapes OK  seg={tuple(seg.shape)}  '
          f'hidden={tuple(new_hidden[0].shape)}  rnn_layers={net.rnn_layers}')
    return net


def check_identity_to_stock(kind):
    """N2/N3 must compute EXACTLY stock CB_TA while their gate is zero."""
    net = _build(kind).eval()
    # Instantiating the ORIGINAL ConditionalUNet requires temporarily putting
    # it back as network.ConditionalUNet: its __init__ calls
    # `super(ConditionalUNet, self)`, a two-arg super() that resolves the class
    # by GLOBAL NAME in network.py at call time. While the patched class owns
    # that name, the stock instance is not a subtype of it and construction
    # dies with "obj must be an instance or subtype of type". This is the same
    # trap the extensions/hooks patch files document as the reason they are
    # standalone modules rather than subclasses.
    saved = cpjku_network.ConditionalUNet
    cpjku_network.ConditionalUNet = _StockUNet
    try:
        torch.manual_seed(0)
        stock = _StockUNet(CONFIG).eval()
    finally:
        cpjku_network.ConditionalUNet = saved

    shared = {k: v for k, v in net.state_dict().items() if k in stock.state_dict()}
    missing = set(stock.state_dict()) - set(shared)
    assert not missing, f'{kind}: stock keys absent from variant (warm start would break): {sorted(missing)}'
    stock.load_state_dict(shared, strict=True)

    score, perf = _inputs()
    with torch.no_grad():
        a = net(score=score, perf=perf, hidden=_zeros_hidden(net))['segmentation']
        b = stock(score=score, perf=perf,
                  hidden=(torch.zeros(1, BS, net.rnn_size),
                          torch.zeros(1, BS, net.rnn_size)))['segmentation']
    diff = (a - b).abs().max().item()
    assert diff < 1e-6, f'{kind}: NOT identical to stock at init (max|diff|={diff:.3e})'
    print(f'  [{kind}] identical to stock CB_TA at init (max|diff|={diff:.2e}) '
          f'-> warm start from B1a is exact')


def check_multi_chunk_and_slot_ops(kind):
    """Chunk carry-over + the two state edits iterate_dataset performs."""
    net = _build(kind).eval()
    hidden = _zeros_hidden(net)
    score, perf = _inputs()
    with torch.no_grad():
        for chunk in range(3):
            hidden = net(score=score, perf=perf, hidden=hidden)['hidden']
            hidden = (hidden[0].detach(), hidden[1].detach())
            assert torch.isfinite(hidden[0]).all() and torch.isfinite(hidden[1]).all(), \
                f'{kind}: non-finite state after chunk {chunk}'

        # per-slot reset (new piece starts in batch slot 0)
        hidden[0][:, 0] = 0
        hidden[1][:, 0] = 0
        out = net(score=score, perf=perf, hidden=hidden)
        assert torch.isfinite(out['segmentation']).all(), f'{kind}: NaN after slot reset'

        # slot drop (a piece finished): iterate_dataset concatenates around dim 1
        hidden = out['hidden']
        idx = 1
        h0 = torch.cat((hidden[0][:, :idx], hidden[0][:, idx + 1:]), dim=1)
        h1 = torch.cat((hidden[1][:, :idx], hidden[1][:, idx + 1:]), dim=1)
        s1, p1 = _inputs(bs=1)
        out = net(score=s1, perf=p1, hidden=(h0, h1))
        assert out['segmentation'].shape == (SEQ_LEN * 1, 1, H, W)
        assert torch.isfinite(out['segmentation']).all(), f'{kind}: NaN after slot drop'
    print(f'  [{kind}] multi-chunk carry, per-slot reset and slot drop all OK')


def check_gradients(kind, new_prefixes):
    net = _build(kind).train()
    score, perf = _inputs()
    out = net(score=score, perf=perf, hidden=_zeros_hidden(net))
    out['segmentation'].mean().backward()

    touched = []
    for name, p in net.named_parameters():
        if any(name.startswith(pref) for pref in new_prefixes):
            assert p.grad is not None, f'{kind}: {name} received no gradient'
            touched.append((name, float(p.grad.abs().sum())))
    assert touched, f'{kind}: no parameters matched {new_prefixes}'
    live = [n for n, g in touched if g > 0]
    assert live, (f'{kind}: every new parameter has an all-zero gradient -- the module '
                  f'can never train. Params: {[n for n, _ in touched]}')
    print(f'  [{kind}] gradients reach {len(live)}/{len(touched)} new tensors '
          f'(e.g. {live[0]})')


def check_branch_unblocks(kind, new_prefixes):
    """The property that actually matters for a silenced-at-init branch.

    Silencing a branch so it starts as exact identity necessarily zeroes the
    gradient of everything downstream of the silencing point on step 0. That
    is only acceptable if it UNBLOCKS: after a single optimiser step the whole
    branch must be receiving gradient. If it does not, the module is dead
    weight for the entire run and the experiment silently measures nothing but
    B1a. So take one real step and re-check every new parameter."""
    net = _build(kind).train()
    params = [p for n, p in net.named_parameters()
              if any(n.startswith(pref) for pref in new_prefixes)]
    opt = torch.optim.Adam(params, lr=1e-2)
    score, perf = _inputs()

    # Carry `hidden` across steps exactly as iterate_dataset does, rather than
    # re-zeroing it. Both because that is what training looks like, and
    # because a zeroed state means an EMPTY memory bank -- and a LayerNorm
    # over an all-zero input has an identically zero gradient w.r.t. its
    # weight ((0-0)/sigma * w multiplies nothing), which would look like a
    # dead parameter while being an artefact of the probe.
    hidden = _zeros_hidden(net)
    dead_now = None
    for _ in range(3):
        opt.zero_grad()
        out = net(score=score, perf=perf, hidden=hidden)
        out['segmentation'].mean().backward()
        dead_now = [n for n, p in net.named_parameters()
                    if any(n.startswith(pref) for pref in new_prefixes)
                    and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
        opt.step()
        hidden = tuple(t.detach() for t in out['hidden'])

    assert not dead_now, (
        f'{kind}: after one optimiser step these new parameters STILL get zero '
        f'gradient -- the branch is permanently dead and the run would just '
        f'reproduce B1a: {sorted(dead_now)}')
    n_new = len(params)
    print(f'  [{kind}] branch unblocks: all {n_new} new tensors receive gradient '
          f'after one step')


def main():
    torch.set_num_threads(1)
    print('N1 long_context (replaces the LSTM)')
    check_shapes_and_state('long_context')
    check_multi_chunk_and_slot_ops('long_context')
    check_gradients('long_context', ('rnn.',))

    print('N2 memory_retrieval (LSTM kept + gated retrieval)')
    check_shapes_and_state('memory_retrieval')
    check_identity_to_stock('memory_retrieval')
    check_multi_chunk_and_slot_ops('memory_retrieval')
    check_gradients('memory_retrieval', ('mem_read.',))
    check_branch_unblocks('memory_retrieval', ('mem_read.',))

    print('N3 belief (LSTM kept + gated Bayes filter)')
    check_shapes_and_state('belief')
    check_identity_to_stock('belief')
    check_multi_chunk_and_slot_ops('belief')
    check_gradients('belief', ('belief_filter.',))
    check_branch_unblocks('belief', ('belief_filter.',))

    print('\nALL TEMPORAL-ARCH SMOKE TESTS PASSED')


if __name__ == '__main__':
    main()
