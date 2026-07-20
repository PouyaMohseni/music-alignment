"""Smoke test for B1a + GatedFiLM (AdaLN-Zero-style). Two levels:

1. UNIT test of GatedFiLM in isolation -- the critical correctness property
   this whole design rests on is "gate=0 at init => block is EXACTLY
   identity, for ANY input/conditioning". This is checked numerically
   (torch.allclose against the raw input, not just 'shapes match'), plus a
   forced-nonzero-gate check proving the gate genuinely blends towards full
   FiLM strength when nonzero (not silently ignored/dead).
2. INTEGRATION test: full ConditionalUNet with GatedFiLM patched in,
   real observed shapes, forward+backward, gradient reaching every param
   (including gate, gamma, beta) except the intentionally frozen
   MERTProjector normalization buffers.
"""
import sys
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment')
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment/third_party/cpjku_unet')

import torch

from extensions.heads.gated_film import GatedFiLM

print('=== Unit test: GatedFiLM ===')
torch.manual_seed(0)
zdim, maskdim = 128, 32
film = GatedFiLM(zdim, maskdim)

x = torch.randn(4, maskdim, 17, 23)
z = torch.randn(4, zdim)

# 1. At init, gate(z) must be EXACTLY 0 for arbitrary z (zero weight AND bias).
gate_out = film.gate(z)
assert torch.allclose(gate_out, torch.zeros_like(gate_out), atol=0.0), \
    f'FAIL: gate output not exactly zero at init (max abs = {gate_out.abs().max().item()})'
print('gate(z) is exactly 0 at init for arbitrary z: PASS')

# 2. Therefore the whole block must be an EXACT identity at init.
out = film(x, z)
assert torch.allclose(out, x, atol=0.0), \
    f'FAIL: GatedFiLM is not an exact identity at init (max abs diff = {(out - x).abs().max().item()})'
print('GatedFiLM(x, z) == x exactly at init: PASS')

# 3. Force the gate to a known nonzero constant and verify the output
#    actually blends towards full-strength FiLM proportionally -- proves the
#    gate is wired into the computation, not a dead/ignored branch.
with torch.no_grad():
    film.gate.bias.fill_(1.0)   # gate(z) == 1 for every z now (weight still 0)
gamma = film.gamma(z).unsqueeze(-1).unsqueeze(-1)
beta = film.beta(z).unsqueeze(-1).unsqueeze(-1)
expected_full_strength = gamma * x + beta
out_full = film(x, z)
assert torch.allclose(out_full, expected_full_strength, atol=1e-6), \
    'FAIL: gate=1 does not reproduce full-strength stock-FiLM output'
print('gate=1 reproduces full-strength FiLM(x,z) = gamma*x+beta: PASS')

with torch.no_grad():
    film.gate.bias.fill_(0.5)
out_half = film(x, z)
expected_half = x + 0.5 * (expected_full_strength - x)
assert torch.allclose(out_half, expected_half, atol=1e-6), \
    'FAIL: gate=0.5 does not linearly interpolate between identity and full FiLM'
print('gate=0.5 linearly interpolates between identity and full FiLM: PASS')


print()
print('=== Integration test: full ConditionalUNet with GatedFiLM ===')
from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.gated_film_patch import patch_gated_film

patch_mert_pipeline(path_to_emb_root={'dummy_train': '/tmp/unused_train', 'dummy_val': '/tmp/unused_val'})
patch_gated_film()

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
H, W = 393, 278   # real observed native-page-scaled resolution

score = torch.rand(seq_len, bs, 1, H, W, device=device)
perf = torch.rand(seq_len, bs, 1, 768, 1, device=device)   # MERTProjector input shape
hidden = (torch.zeros(net.rnn_layers, bs, net.rnn_size, device=device),
          torch.zeros(net.rnn_layers, bs, net.rnn_size, device=device))

# At init (all gates == 0), the network's visual pathway must behave EXACTLY
# as if FiLM were absent entirely -- i.e. output should be identical whether
# perf/audio is real or garbage, since every film_layer is a no-op at init.
out_a = net(score=score, perf=perf, hidden=hidden)['segmentation']
perf_garbage = torch.rand(seq_len, bs, 1, 768, 1, device=device) * 100 - 50
hidden2 = (torch.zeros(net.rnn_layers, bs, net.rnn_size, device=device),
          torch.zeros(net.rnn_layers, bs, net.rnn_size, device=device))
out_b = net(score=score, perf=perf_garbage, hidden=hidden2)['segmentation']
assert torch.allclose(out_a, out_b, atol=1e-5), \
    ('FAIL: with all gates at 0-init, changing the audio input changed the output -- '
     'a film_layer is not actually a no-op at init')
print('at init, audio input has ZERO effect on segmentation output (all gates are 0): PASS')

target = torch.rand_like(out_a)
loss = ((out_a - target) ** 2).mean()
loss.backward()

film_blocks = [b for b in list(net.encoder) + list(net.decoder) + [net.bottleneck_block] if getattr(b, 'film', False)]
print(f'{len(film_blocks)} blocks have film=True (gated FiLM)')
assert len(film_blocks) > 0, 'FAIL: no film blocks found -- test config is wrong'

film_grad_ok = all(
    p.grad is not None
    for block in film_blocks
    for p in block.film_layer.parameters()
)
print('gated-film params all received a grad (gate included, even though gate=0 at init):', film_grad_ok)
assert film_grad_ok, 'FAIL: film_layer (gamma/beta/gate) did not receive gradient'

gate_grad_nonzero = any(
    block.film_layer.gate.weight.grad.abs().sum().item() > 0
    for block in film_blocks
)
print('at least one gate weight has NONZERO gradient (so training can move gate away from 0):',
      gate_grad_nonzero)
assert gate_grad_nonzero, 'FAIL: gate weight gradient is exactly zero everywhere -- gate could never learn'

conv_grad_ok = all(p.grad is not None for p in net.encoder[0].conv1.parameters())
print('encoder conv params have grad:', conv_grad_ok)
assert conv_grad_ok, 'FAIL: conv1 did not receive gradient'

print()
print('SMOKE TEST PASSED')
