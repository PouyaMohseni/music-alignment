"""Smoke test for MERT+DINOv2 cross-attention: uses SYNTHETIC grid data
(doesn't need the real DINOv2 precompute to be done) -- verifies
ConditionalUNetMERTDINOv2CrossAttn's shape-matching logic (_maxpool_out
chain, visual_neck adapters for the content/skip pathway, TokenCrossAttn
FiLM's raw-token K/V pathway, decoder skip-connection consumption) before
spending any real training time or waiting on real precompute data.
"""
import sys
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment')
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment/third_party/cpjku_unet')

import torch

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.mert_dinov2_cross_attention_patch import _build_mert_dinov2_crossattn_network
from audio_conditioned_unet import network as cpjku_network

patch_mert_pipeline(path_to_emb_root={'dummy_train': '/tmp/unused_train', 'dummy_val': '/tmp/unused_val'})

net_config = {
    'film_layers': [2, 3, 4, 5, 6, 7, 8],
    'n_encoder_layers': 4,
    'n_filters_start': 8,
    'rnn_size': 128,
    'rnn_layer': 1,
    'use_lstm': True,
    'audio_encoder': 'MERTProjector',
    'spec_enc': 32,
}

ConditionalUNetMERTDINOv2CrossAttn = _build_mert_dinov2_crossattn_network(cpjku_network)
net = ConditionalUNetMERTDINOv2CrossAttn(net_config)
print(f'params: {sum(p.numel() for p in net.parameters()):,}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

seq_len, bs_pieces = 4, 2
H, W = 393, 278          # real observed native-page-scaled resolution
n_rows, n_cols = 6, 4    # real observed tile grid for a 1181x835 page at tile_size=224

score = torch.rand(seq_len, bs_pieces, 1, H, W, device=device)
perf = torch.rand(seq_len, bs_pieces, 1, 768, 1, device=device)   # MERTProjector input shape
hidden = (torch.zeros(net.rnn_layers, bs_pieces, net.rnn_size, device=device),
          torch.zeros(net.rnn_layers, bs_pieces, net.rnn_size, device=device))
visual_grid = torch.rand(bs_pieces, n_rows, n_cols, 768, device=device, requires_grad=False)

out = net(score=score, perf=perf, hidden=hidden, visual_grid=visual_grid)
seg = out['segmentation']
print('segmentation shape:', seg.shape, 'expected:', (seq_len * bs_pieces, 1, H, W))
assert seg.shape == (seq_len * bs_pieces, 1, H, W), 'FAIL: output shape mismatch'

target = torch.rand_like(seg)
loss = ((seg - target) ** 2).mean()
loss.backward()

neck_grad_ok = all(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in net.visual_neck.parameters())
film_blocks = [b for b in list(net.decoder) + [net.bottleneck_block] if getattr(b, 'film', False)]
print(f'{len(film_blocks)} blocks have film=True (token cross-attention FiLM)')
film_grad_ok = all(
    p.grad is not None and p.grad.abs().sum().item() > 0
    for block in film_blocks
    for p in block.film_layer.parameters()
)
# net.perf_encoder.means/.stds are requires_grad=False (frozen normalization
# stats, kept for CBEncoder-interface parity) -- only .enc is trainable.
mert_grad_ok = all(p.grad is not None for p in net.perf_encoder.enc.parameters())

print('visual_neck params all have nonzero grad:', neck_grad_ok)
print('token cross-attention film params all have nonzero grad:', film_grad_ok)
print('MERTProjector params have grad:', mert_grad_ok)
assert neck_grad_ok, 'FAIL: visual_neck did not receive gradient'
assert film_grad_ok, 'FAIL: film_layer did not receive gradient'
assert mert_grad_ok, 'FAIL: perf_encoder (MERTProjector) did not receive gradient'

print('SMOKE TEST PASSED')
