"""Smoke test for the DINOv2 full-encoder-replacement network, using
SYNTHETIC grid data (doesn't need the real precompute to be done) --
verifies the shape-matching logic (_maxpool_out chain, neck adapters,
decoder skip-connection consumption) is actually correct before spending
any real training time or waiting on real DINOv2 data.
"""
import sys
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment')
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment/third_party/cpjku_unet')

import torch

from extensions.hooks.dinov2_full_encoder_patch import _build_dinov2_network, _maxpool_out
from audio_conditioned_unet import network as cpjku_network

net_config = {
    'film_layers': [2, 3, 4, 5, 6, 7, 8],
    'n_encoder_layers': 4,
    'n_filters_start': 8,
    'rnn_size': 128,
    'rnn_layer': 1,
    'use_lstm': True,
    'audio_encoder': 'CBEncoder',
    'spec_enc': 32,
}

ConditionalUNetDINOv2Visual = _build_dinov2_network(cpjku_network, dinov2_root='/tmp/unused')
net = ConditionalUNetDINOv2Visual(net_config)
print(f'params: {sum(p.numel() for p in net.parameters()):,}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

seq_len, bs_pieces = 4, 2
H, W = 393, 278   # real observed native-page-scaled resolution
n_rows, n_cols = 6, 4   # real observed tile grid for a 1181x835 page at tile_size=224

score = torch.rand(seq_len, bs_pieces, 1, H, W, device=device)
perf = torch.rand(seq_len, bs_pieces, 1, 78, 40, device=device)   # CBEncoder input shape
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
decoder_grad_ok = all(p.grad is not None for p in net.decoder.parameters() if p.requires_grad)
print('visual_neck params all have nonzero grad:', neck_grad_ok)
print('decoder params all have grad:', decoder_grad_ok)
assert neck_grad_ok, 'FAIL: visual_neck did not receive gradient'
assert decoder_grad_ok, 'FAIL: decoder did not receive gradient'

print('SMOKE TEST PASSED')
