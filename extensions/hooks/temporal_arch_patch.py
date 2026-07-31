"""Network patches for the three temporal-architecture experiments N1/N2/N3.

They live in one module because they are the SAME edit to ConditionalUNet --
keep the visual encoder, decoder, bottleneck, FiLM and conv_out byte-for-byte
identical to CB_TA, and change only how temporal state is carried and used --
differing purely in which temporal component is installed. Splitting them
into three files would triplicate ~90 lines of copied ConditionalUNet
skeleton, which is exactly the drift risk this project has already been
bitten by; the per-experiment reasoning lives in the three head modules'
docstrings (extensions/heads/{long_context_temporal,gated_memory_retrieval,
belief_propagation}.py).

  N1 patch_long_context_temporal()  -- REPLACE the LSTM with a two-tier
                                       memory Transformer.
  N2 patch_gated_memory_retrieval() -- KEEP the LSTM verbatim, add a
                                       zero-init-gated retrieval read.
  N3 patch_gated_belief_propagation() -- KEEP the LSTM verbatim, add a
                                       zero-init-gated Bayes filter on the
                                       output heatmap.

Shared design decisions, all forced by measured results in this project:
  * The conv visual encoder is untouched. Replacing it with DINOv2 features
    scored 6.9-9.2% (V_dinov2_full_encoder) -- the coarse tile grid destroys
    the spatial resolution the task needs.
  * FiLM is untouched. Every FiLM replacement lost ground versus plain FiLM
    (spatial 44.3%, cross-attention 71.1%, gated 82.9%, vs B1a 89.2%).
  * N2/N3 are zero-init-gated additions, so at initialisation they compute
    EXACTLY B1a and can be warm-started from its converged checkpoint --
    the pattern behind the only result that has beaten B1a (B3, 89.8%).

NOT a subclass of ConditionalUNet: its __init__ calls
`super(ConditionalUNet, self).__init__()`, an explicit two-arg super() that
resolves the class by GLOBAL NAME in network.py at call time, so once that
name is monkey-patched a subclass recurses into itself. Same reason
cross_attention_film_patch.py gives for ConditionalUNetBlock. These are
independent nn.Modules using Python 3 zero-arg super().
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_LONG_CONTEXT = 'long_context'
_MEMORY_RETRIEVAL = 'memory_retrieval'
_BELIEF = 'belief'


def _build_network(cpjku_network_module, kind: str, opts: dict):
    audio_encoder = cpjku_network_module.audio_encoder
    initialize_weights = cpjku_network_module.initialize_weights
    ConditionalUNetBlock = cpjku_network_module.ConditionalUNetBlock

    class ConditionalUNetTemporal(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.n_encoder_layers = config.get('n_encoder_layers', 4)
            self.n_filters_start = config.get('n_filters_start', 8)
            self.use_lstm = config.get('use_lstm', False)
            self.max_channel = 128
            self.rnn_size = config.get('rnn_size', 512)
            self.spec_enc = config.get('spec_enc', 512)
            lstm_layers = config.get('rnn_layer', 1)

            if not self.use_lstm:
                raise RuntimeError(
                    'temporal_arch_patch requires --use_lstm: all three variants '
                    'replace or augment the recurrent temporal path, which does not '
                    'exist in the no-LSTM (per-frame nn.Linear) configuration.')
            if kind in (_MEMORY_RETRIEVAL, _BELIEF) and lstm_layers != 1:
                raise RuntimeError(
                    f'{kind} packs the LSTM state into hidden[i][0:1] and therefore '
                    f'assumes a single LSTM layer, got rnn_layer={lstm_layers}.')

            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()
            self.perf_encoder = getattr(audio_encoder, config['audio_encoder'])(self.spec_enc)

            # --- temporal component (the only structural difference) --------
            self.kind = kind
            if kind == _LONG_CONTEXT:
                from extensions.heads.long_context_temporal import LongContextTemporalCore
                self.rnn = LongContextTemporalCore(
                    d_in=self.spec_enc, d_model=self.rnn_size,
                    n_layers=opts.get('n_layers', 2), n_heads=opts.get('n_heads', 8),
                    n_fine=opts.get('n_fine', 64), n_comp=opts.get('n_comp', 192),
                    pool=opts.get('pool', 16))
                self.rnn_layers = self.rnn.state_depth
            else:
                # Named `rnn` with stock nn.LSTM parameter names
                # (rnn.weight_ih_l0, ...) so B1a's checkpoint restores it exactly.
                self.rnn = nn.LSTM(self.spec_enc, hidden_size=self.rnn_size,
                                   num_layers=1, batch_first=False)
                if kind == _MEMORY_RETRIEVAL:
                    from extensions.heads.gated_memory_retrieval import GatedMemoryRetrieval
                    self.mem_read = GatedMemoryRetrieval(
                        d_model=self.rnn_size, n_heads=opts.get('n_heads', 8),
                        n_mem=opts.get('n_mem', 192), pool=opts.get('pool', 16))
                    self.rnn_layers = self.mem_read.state_depth
                else:
                    from extensions.heads.belief_propagation import GatedBeliefPropagation
                    self.belief_filter = GatedBeliefPropagation(
                        d_model=self.rnn_size,
                        belief_h=opts.get('belief_h', 16), belief_w=opts.get('belief_w', 64))
                    self.rnn_layers = self.belief_filter.state_depth

            # --- everything below is CB_TA, unchanged ------------------------
            film_layers = config['film_layers']
            for i in range(1, self.n_encoder_layers + 1):
                if i == 1:
                    in_, out_ = 1, min(self.n_filters_start, self.max_channel)
                else:
                    in_ = min(self.n_filters_start * (2 ** (i - 2)), self.max_channel)
                    out_ = min(self.n_filters_start * (2 ** (i - 1)), self.max_channel)
                self.encoder.append(
                    ConditionalUNetBlock(in_, out_, self.rnn_size, film=i in film_layers))
                self.decoder.append(ConditionalUNetBlock(
                    out_, out_, self.rnn_size,
                    film=2 * (self.n_encoder_layers + 1) - i in film_layers,
                    up_in_channels=min(out_ * 2, self.max_channel),
                    up_sample=True, down_sample=False))

            self.bottleneck_block = ConditionalUNetBlock(
                min(self.n_filters_start * (2 ** (self.n_encoder_layers - 1)), self.max_channel),
                min(self.n_filters_start * (2 ** self.n_encoder_layers), self.max_channel),
                self.rnn_size, film=self.n_encoder_layers + 1 in film_layers, down_sample=False)
            self.conv_out = nn.Conv2d(self.n_filters_start, 1, kernel_size=(1, 1))

            self.first_execution = True
            self.apply(initialize_weights)

        def forward(self, score, perf, hidden):
            seq_len, bs, c, h, w = score.shape
            x = score.view(seq_len * bs, c, h, w)

            perf = self.perf_encoder(perf).view(seq_len, bs, -1)

            belief_state = belief_valid = None
            if self.kind == _LONG_CONTEXT:
                perf, hidden = self.rnn(perf, hidden)
            elif self.kind == _MEMORY_RETRIEVAL:
                lstm_hidden, bank, bank_valid = self.mem_read.split_state(hidden)
                perf, lstm_hidden = self.rnn(perf, lstm_hidden)
                perf = self.mem_read.read(perf, bank, bank_valid)
                bank, bank_valid = self.mem_read.update(bank, bank_valid, perf)
                hidden = self.mem_read.pack_state(lstm_hidden, bank, bank_valid)
            else:
                lstm_hidden, belief_state, belief_valid = self.belief_filter.split_state(hidden)
                perf, lstm_hidden = self.rnn(perf, lstm_hidden)

            perf = perf.reshape(seq_len * bs, -1)

            residuals = []
            for i in range(self.n_encoder_layers):
                res, x = self.encoder[i](x, perf)
                residuals.append(res)
                if self.first_execution:
                    print('down', x.shape)

            x = self.bottleneck_block(x, perf)
            if self.first_execution:
                print('bottleneck', x.shape)

            for i in range(self.n_encoder_layers)[::-1]:
                x = self.decoder[i](x, perf, residuals[i])
                if self.first_execution:
                    print('up', x.shape)

            x = self.conv_out(x)

            if self.kind == _BELIEF:
                x, belief_state, belief_valid = self.belief_filter(
                    x, belief_state, belief_valid, seq_len, bs)
                hidden = self.belief_filter.pack_state(
                    lstm_hidden, belief_state, belief_valid)

            if self.first_execution:
                print('out', x.shape)
                self.first_execution = False

            return {'segmentation': torch.sigmoid(x), 'hidden': hidden}

    return ConditionalUNetTemporal


def _patch_initialize_weights(cpjku_network):
    """ConditionalUNet.__init__ ends with self.apply(initialize_weights),
    which orthogonal-inits every nn.Linear/nn.Conv2d it finds -- including a
    branch's deliberately zero-initialised final layer, silently undoing it
    AFTER construction. That exact bug was already caught once in this repo
    (GatedFiLM's gate leaked non-zero at 'init'). Honour the `_zero_init`
    tag instead. Idempotent, so repeated patching cannot re-wrap."""
    if getattr(cpjku_network.initialize_weights, '_honours_zero_init', False):
        return
    _orig = cpjku_network.initialize_weights

    def initialize_weights_preserving_zero_init(m):
        if getattr(m, '_zero_init', False):
            nn.init.zeros_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
            return
        _orig(m)

    initialize_weights_preserving_zero_init._honours_zero_init = True
    cpjku_network.initialize_weights = initialize_weights_preserving_zero_init


def _install(kind: str, opts: dict, label: str):
    from audio_conditioned_unet import network as cpjku_network
    _patch_initialize_weights(cpjku_network)
    cpjku_network.ConditionalUNet = _build_network(cpjku_network, kind, opts)
    print(f'[temporal_arch_patch] {label}', flush=True)


def patch_long_context_temporal(**opts):
    _install(_LONG_CONTEXT, opts,
             'N1: replaced the LSTM with a two-tier (fine + compressed) memory '
             'Transformer temporal core; visual encoder and FiLM unchanged')


def patch_gated_memory_retrieval(**opts):
    _install(_MEMORY_RETRIEVAL, opts,
             'N2: kept the LSTM verbatim and added a zero-init-gated, lag-aware '
             'retrieval read over the piece\'s own compressed history')


def patch_gated_belief_propagation(**opts):
    _install(_BELIEF, opts,
             'N3: kept the LSTM verbatim and added a zero-init-gated differentiable '
             'Bayes filter (learned 2D transition + uniform floor) as a log-prior '
             'on the output heatmap')
