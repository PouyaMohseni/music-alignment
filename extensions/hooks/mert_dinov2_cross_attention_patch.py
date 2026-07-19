"""General cross-attention fusion of MERT (audio) and DINOv2 (visual): the
'actual new method' requested as a follow-up to V-DINOv2 (drop-in visual-
encoder swap, FiLM untouched) and B1a-cross-attention (FiLM replaced, but
K/V = the from-scratch encoder's own feature map). This is the version that
changes BOTH sides AND the fusion mechanism together:

- Audio: MERTProjector (extensions/audio_encoders/mert_projector.py), same
  frozen-MERT load_piece patch as B1a (extensions/hooks/mert_patch.py).
- Visual: DINOv2 tile grid. Reuses DINOv2VisualNeck
  (extensions/heads/dinov2_visual_neck.py) for the per-stage interpolated
  residuals/bottleneck-input CONTENT pathway -- decoder skip connections
  still need a spatially-shaped tensor at each resolution, identical
  plumbing to V-DINOv2 -- but ALSO exposes the RAW (un-interpolated) DINOv2
  patch-token grid globally to every FiLM site.
- Fusion: TokenCrossAttentionFiLM (extensions/heads/cross_attention_film.py)
  replaces vanilla FiLM at every decoder/bottleneck block: the MERT audio
  embedding is the attention QUERY, the raw DINOv2 patch tokens (genuine
  per-patch tokens, not an interpolated feature map) are KEY/VALUE. This is
  literally an alignment mechanism -- audio asks "which patch of the score
  is relevant right now" -- replacing FiLM's context-blind uniform
  broadcast.

Reuses iterate_dataset_visual + the DINOv2 grid loader/cache from
extensions/hooks/dinov2_full_encoder_patch.py unchanged for visual_grid
batching. MERT's own precomputed-embedding loading goes through the
unchanged mert_patch.py load_piece patch (orthogonal to visual_grid --
MERT's `perf` tensor and the DINOv2 `visual_grid` tensor are independent
inputs to network.forward).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _maxpool_out(h: int, w: int) -> tuple[int, int]:
    """Matches nn.MaxPool2d(kernel_size=2, stride=2)'s exact output formula
    (padding=0, dilation=1): floor((size - 2) / 2) + 1."""
    return (h - 2) // 2 + 1, (w - 2) // 2 + 1


class _TokenCrossAttnBlock(nn.Module):
    """Structurally identical to audio_conditioned_unet.network.
    ConditionalUNetBlock (same conv1/conv2/norm1/norm2/up_conv/max_pool),
    except the FiLM step is TokenCrossAttentionFiLM and forward() threads an
    extra `visual_tokens` tensor (the raw DINOv2 patch grid, shared by every
    block in the network) as cross-attention K/V. A fully independent
    nn.Module (not a subclass of the original block) -- see
    cross_attention_film_patch.py's docstring for why subclassing the stock
    block is unsafe (its __init__ calls super(ConditionalUNetBlock, self),
    an explicit super() that resolves the class name as a module global at
    call time and breaks under monkey-patching)."""

    def __init__(self, in_channels, out_channels, spec_out, token_dim=768, film=True,
                 down_sample=True, up_sample=False, up_in_channels=1, padding=1, n_heads=4):
        super().__init__()
        self.up_sample = up_sample
        self.down_sample = down_sample
        self.film = film

        if self.up_sample:
            self.up_conv = nn.Sequential(nn.Upsample(scale_factor=2),
                                         nn.Conv2d(up_in_channels, in_channels, kernel_size=1, stride=1))
        if self.down_sample:
            self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)

        if self.film:
            from extensions.heads.cross_attention_film import TokenCrossAttentionFiLM
            self.film_layer = TokenCrossAttentionFiLM(spec_out, out_channels, token_dim=token_dim, n_heads=n_heads)

    def forward(self, x, spec, visual_tokens, residual=None):
        if self.up_sample:
            x = self.up_conv(x)
            if residual is not None:
                diffY = residual.size()[2] - x.size()[2]
                diffX = residual.size()[3] - x.size()[3]
                x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
                x = x + residual

        x = F.elu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        if self.film:
            x = self.film_layer(x, spec, visual_tokens)

        x = F.elu(x)

        if self.down_sample:
            return x, self.max_pool(x)
        else:
            return x


def _build_mert_dinov2_crossattn_network(cpjku_network_module, d_dinov2: int = 768):
    from extensions.heads.dinov2_visual_neck import DINOv2VisualNeck

    audio_encoder = cpjku_network_module.audio_encoder
    initialize_weights = cpjku_network_module.initialize_weights

    class ConditionalUNetMERTDINOv2CrossAttn(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.n_encoder_layers = config.get('n_encoder_layers', 4)
            self.n_filters_start = config.get('n_filters_start', 8)
            self.use_lstm = config.get('use_lstm', False)
            self.max_channel = 128

            self.decoder = nn.ModuleList()
            self.rnn_size = config.get('rnn_size', 512)
            self.rnn_layers = config.get('rnn_layer', 1)
            self.spec_enc = config.get('spec_enc', 512)

            self.perf_encoder = getattr(audio_encoder, config['audio_encoder'])(self.spec_enc)

            if self.use_lstm:
                self.rnn = nn.LSTM(self.spec_enc, hidden_size=self.rnn_size,
                                   num_layers=self.rnn_layers, batch_first=False)
            else:
                self.fc = nn.Linear(self.spec_enc, self.rnn_size)

            film_layers = config['film_layers']
            stage_channels = []
            for i in range(1, self.n_encoder_layers + 1):
                if i == 1:
                    out_ = min(self.n_filters_start, self.max_channel)
                else:
                    out_ = min(self.n_filters_start * (2 ** (i - 1)), self.max_channel)
                stage_channels.append(out_)

                dec_block = _TokenCrossAttnBlock(
                    out_, out_, self.rnn_size, token_dim=d_dinov2,
                    film=2 * (self.n_encoder_layers + 1) - i in film_layers,
                    up_in_channels=min(out_ * 2, self.max_channel), up_sample=True, down_sample=False)
                self.decoder.append(dec_block)

            bottleneck_channels = min(self.n_filters_start * (2 ** self.n_encoder_layers), self.max_channel)
            self.bottleneck_block = _TokenCrossAttnBlock(
                stage_channels[-1], bottleneck_channels, self.rnn_size, token_dim=d_dinov2,
                film=self.n_encoder_layers + 1 in film_layers, down_sample=False)

            self.conv_out = nn.Conv2d(self.n_filters_start, 1, kernel_size=(1, 1))

            self.visual_neck = DINOv2VisualNeck(stage_channels=stage_channels,
                                                bottleneck_channels=stage_channels[-1], d_dinov2=d_dinov2)

            self.first_execution = True
            self.apply(initialize_weights)

        def forward(self, score, perf, hidden, visual_grid=None):
            if visual_grid is None:
                raise RuntimeError('ConditionalUNetMERTDINOv2CrossAttn requires visual_grid '
                                   '(bs, n_rows, n_cols, 768) -- use iterate_dataset_visual, '
                                   'not the stock iterate_dataset.')

            seq_len, bs, c, h, w = score.shape

            perf = self.perf_encoder(perf)
            if self.use_lstm:
                perf = perf.view(seq_len, bs, -1)
                perf, hidden = self.rnn(perf, hidden)
                perf = perf.view(seq_len * bs, -1)
            else:
                perf = F.elu(self.fc(perf))

            stage_sizes = []
            cur_h, cur_w = h, w
            for _ in range(self.n_encoder_layers):
                stage_sizes.append((cur_h, cur_w))
                cur_h, cur_w = _maxpool_out(cur_h, cur_w)
            bottleneck_size = (cur_h, cur_w)

            # visual_grid: (bs_pieces, n_rows, n_cols, 768), one per BATCH SLOT
            # (constant across that slot's seq_len frames) -- expand to match
            # the (seq_len*bs_pieces) frame-batch dimension used everywhere else.
            grid = visual_grid.unsqueeze(0).expand(seq_len, -1, -1, -1, -1)
            grid = grid.reshape(seq_len * visual_grid.shape[0], *visual_grid.shape[1:])   # (SB, n_rows, n_cols, 768)

            residuals, x = self.visual_neck(grid, stage_sizes, bottleneck_size)
            raw_tokens = grid.reshape(grid.shape[0], -1, grid.shape[-1])   # (SB, n_tokens, 768)

            if self.first_execution:
                for r in residuals:
                    print('visual_neck residual', r.shape)
                print('raw_tokens', raw_tokens.shape)

            x = self.bottleneck_block(x, perf, raw_tokens)
            if self.first_execution:
                print('bottleneck', x.shape)

            for i in range(self.n_encoder_layers)[::-1]:
                x = self.decoder[i](x, perf, raw_tokens, residuals[i])
                if self.first_execution:
                    print('up', x.shape)

            x = self.conv_out(x)
            if self.first_execution:
                print('out', x.shape)
                self.first_execution = False

            x = torch.sigmoid(x)
            return {'segmentation': x, 'hidden': hidden}

    return ConditionalUNetMERTDINOv2CrossAttn


def patch_mert_dinov2_cross_attention(dinov2_root: str, mert_path_to_emb_root: dict[str, str]):
    from extensions.hooks.mert_patch import patch_mert_pipeline
    patch_mert_pipeline(path_to_emb_root=mert_path_to_emb_root)

    from audio_conditioned_unet import network as cpjku_network
    from audio_conditioned_unet import dataset as cpjku_dataset
    import extensions.hooks.dinov2_full_encoder_patch as _dfp

    _dfp._DINOV2_ROOT = dinov2_root

    cpjku_network.ConditionalUNet = _build_mert_dinov2_crossattn_network(cpjku_network)
    cpjku_dataset.iterate_dataset = _dfp.iterate_dataset_visual

    print(f'[mert_dinov2_cross_attention_patch] MERT audio + raw-DINOv2-token cross-attention '
          f'FiLM replacement (dinov2_root={dinov2_root})', flush=True)
