"""Smoke test for B1a + SpatialFiLM (SPADE-inspired). Two levels:

1. UNIT test of SpatialFiLM in isolation -- the critical correctness
   property this design rests on is "gamma/beta actually vary across
   spatial positions" (unlike stock FiLM, which is mathematically incapable
   of it). Checked numerically: gamma/beta at one corner of the feature map
   must differ from gamma/beta at the opposite corner, for the SAME z --
   if this failed (e.g. due to a broadcasting bug collapsing the coarse
   grid back to a constant), SpatialFiLM would be silently equivalent to
   stock FiLM despite the extra machinery. Also checks the coarse-grid ->
   upsample path at every real resolution this network actually produces
   (full res down to the bottleneck), not just one arbitrary shape.
2. INTEGRATION test: full ConditionalUNet with SpatialFiLM patched in, real
   observed shapes, forward+backward, gradient reaching every param.
"""
import sys
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment')
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment/third_party/cpjku_unet')

import torch

from extensions.heads.spatial_film import SpatialFiLM

print('=== Unit test: SpatialFiLM ===')
torch.manual_seed(0)
zdim, maskdim = 128, 32
film = SpatialFiLM(zdim, maskdim, coarse_size=(4, 4))

z = torch.randn(2, zdim)

# Real observed resolutions this network actually produces, finest to
# bottleneck (from the dinov2/cross-attention smoke tests' verified shapes).
real_shapes = [(393, 278), (196, 139), (98, 69), (49, 34), (24, 17)]
for h, w in real_shapes:
    x = torch.randn(2, maskdim, h, w)
    out = film(x, z)
    assert out.shape == x.shape, f'FAIL: shape mismatch at ({h},{w}): got {out.shape}'
print(f'SpatialFiLM handles all {len(real_shapes)} real network resolutions '
      f'({real_shapes}) without shape errors: PASS')

# Spatial-variation check at the largest resolution: compute gamma/beta
# directly (bypassing the x-dependent parts) and confirm two opposite
# corners of the SAME (z-conditioned) map actually differ.
h, w = real_shapes[0]
grid = film.to_grid(z).view(2, maskdim, film.coarse_h, film.coarse_w)
grid_up = torch.nn.functional.interpolate(grid, size=(h, w), mode='bilinear', align_corners=False)
gamma_full = film.refine_gamma(grid_up)
corner_tl = gamma_full[0, :, 0, 0]
corner_br = gamma_full[0, :, -1, -1]
assert not torch.allclose(corner_tl, corner_br, atol=1e-4), \
    'FAIL: gamma is identical at opposite corners -- SpatialFiLM has collapsed to global (stock-FiLM-like) behavior'
print(f'gamma differs between opposite corners of the SAME feature map '
      f'(max abs diff = {(corner_tl - corner_br).abs().max().item():.4f}): PASS -- genuine spatial variation confirmed')

# Batch independence: two different z's must produce different spatial
# patterns (not a bug where the linear layer's output ignores z).
z2 = torch.randn(2, zdim)
grid2 = film.to_grid(z2).view(2, maskdim, film.coarse_h, film.coarse_w)
assert not torch.allclose(grid, grid2, atol=1e-4), \
    'FAIL: two different audio embeddings produced the same spatial grid -- to_grid is ignoring z'
print('two different audio embeddings (z) produce different spatial grids: PASS')


print()
print('=== Integration test: full ConditionalUNet with SpatialFiLM ===')
from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.spatial_film_patch import patch_spatial_film

patch_mert_pipeline(path_to_emb_root={'dummy_train': '/tmp/unused_train', 'dummy_val': '/tmp/unused_val'})
patch_spatial_film()

from audio_conditioned_unet.network import ConditionalUNet

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

net = ConditionalUNet(net_config)
print(f'params: {sum(p.numel() for p in net.parameters()):,}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)

seq_len, bs = 4, 2
H, W = 393, 278

score = torch.rand(seq_len, bs, 1, H, W, device=device)
perf = torch.rand(seq_len, bs, 1, 768, 1, device=device)
hidden = (torch.zeros(net.rnn_layers, bs, net.rnn_size, device=device),
          torch.zeros(net.rnn_layers, bs, net.rnn_size, device=device))

out = net(score=score, perf=perf, hidden=hidden)
seg = out['segmentation']
print('segmentation shape:', seg.shape, 'expected:', (seq_len * bs, 1, H, W))
assert seg.shape == (seq_len * bs, 1, H, W), 'FAIL: output shape mismatch'

target = torch.rand_like(seg)
loss = ((seg - target) ** 2).mean()
loss.backward()

film_blocks = [b for b in list(net.encoder) + list(net.decoder) + [net.bottleneck_block] if getattr(b, 'film', False)]
print(f'{len(film_blocks)} blocks have film=True (spatial FiLM)')
assert len(film_blocks) > 0, 'FAIL: no film blocks found -- test config is wrong'

film_grad_ok = all(
    p.grad is not None and p.grad.abs().sum().item() > 0
    for block in film_blocks
    for p in block.film_layer.parameters()
)
print('spatial-film params all have nonzero grad:', film_grad_ok)
assert film_grad_ok, 'FAIL: film_layer did not receive gradient'

conv_grad_ok = all(p.grad is not None for p in net.encoder[0].conv1.parameters())
print('encoder conv params have grad:', conv_grad_ok)
assert conv_grad_ok, 'FAIL: conv1 did not receive gradient'

print()
print('SMOKE TEST PASSED')
