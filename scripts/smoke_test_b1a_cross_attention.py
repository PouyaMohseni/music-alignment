"""Smoke test for B1a + cross-attention FiLM: verifies
SpatialCrossAttentionFiLM correctly replaces FiLM inside ConditionalUNet
(unchanged conv encoder/decoder, unchanged MERTProjector interface) and that
gradients flow through both the cross-attention film_layer params and the
rest of the network, before spending any real training time.
"""
import sys
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment')
sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment/third_party/cpjku_unet')

import torch

from extensions.hooks.mert_patch import patch_mert_pipeline
from extensions.hooks.cross_attention_film_patch import patch_cross_attention_film

patch_mert_pipeline(path_to_emb_root={'dummy_train': '/tmp/unused_train', 'dummy_val': '/tmp/unused_val'})
patch_cross_attention_film()

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

out = net(score=score, perf=perf, hidden=hidden)
seg = out['segmentation']
print('segmentation shape:', seg.shape, 'expected:', (seq_len * bs, 1, H, W))
assert seg.shape == (seq_len * bs, 1, H, W), 'FAIL: output shape mismatch'

target = torch.rand_like(seg)
loss = ((seg - target) ** 2).mean()
loss.backward()

film_blocks = [b for b in list(net.encoder) + list(net.decoder) + [net.bottleneck_block] if getattr(b, 'film', False)]
print(f'{len(film_blocks)} blocks have film=True (cross-attention FiLM)')
assert len(film_blocks) > 0, 'FAIL: no film blocks found -- test config is wrong'

film_grad_ok = all(
    p.grad is not None and p.grad.abs().sum().item() > 0
    for block in film_blocks
    for p in block.film_layer.parameters()
)
print('cross-attention film params all have nonzero grad:', film_grad_ok)
assert film_grad_ok, 'FAIL: film_layer did not receive gradient'

conv_grad_ok = all(p.grad is not None for p in net.encoder[0].conv1.parameters())
print('encoder conv params have grad:', conv_grad_ok)
assert conv_grad_ok, 'FAIL: conv1 did not receive gradient'

print('SMOKE TEST PASSED')
