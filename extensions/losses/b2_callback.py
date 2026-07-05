"""B2 aux_loss_fn callback for iterate_dataset_ext. Lazily creates
PitchAuxHead on first call (channel/hidden sizes only known once real
rnn_out/decoder_feature shapes are captured) and registers it with the
optimizer -- same reason as B3/B5: train_model.py's optimizer is built
before iterate_dataset ever runs.

Supervision: BCE against the 88-dim active-pitch multi-hot vector at the
GT-onset frame (pitch_batch, from extensions/hooks/pitch_patch.py), for
BOTH heads (audio_pitch_head from the LSTM state, score_pitch_head from the
FiLM-modulated decoder feature sampled at the GT location) -- both heads
deleted at inference, per CB_TA-Ext.md.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.hooks.position_decoder import center_of_mass_xy
from extensions.heads.pitch_aux_head import PitchAuxHead


def b2_aux_loss(pred, y_batch, rnn_out, decoder_feature, seq_len, bs, network, optimizer, pitch_batch):
    if decoder_feature is None or rnn_out is None:
        raise RuntimeError('B2 requires need_rnn_capture=True and decoder_feature_stage set')
    if pitch_batch is None:
        raise RuntimeError('B2 requires need_pitch_roll=True')

    C = decoder_feature.shape[1]
    rnn_hidden = rnn_out.shape[-1]
    if not hasattr(network, '_ext_b2_pitch_head'):
        head = PitchAuxHead(rnn_hidden, C).to(decoder_feature.device)
        network.add_module('_ext_b2_pitch_head', head)
        optimizer.add_param_group({'params': head.parameters()})
    head = network._ext_b2_pitch_head

    H, W = pred.shape[-2], pred.shape[-1]
    y_grid = y_batch.view(seq_len * bs, H, W)
    gt_xy = center_of_mass_xy(y_grid)   # (seq_len*bs, 2)

    audio_logits, score_logits, _sampled_feature = head(rnn_out, decoder_feature, gt_xy, score_hw=(H, W))

    pitch_target = pitch_batch.reshape(seq_len * bs, -1)   # (seq_len*bs, 88)
    loss = (F.binary_cross_entropy_with_logits(audio_logits, pitch_target)
           + F.binary_cross_entropy_with_logits(score_logits, pitch_target))
    return loss, {'pitch_aux': loss.detach()}
