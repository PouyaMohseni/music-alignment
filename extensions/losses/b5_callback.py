"""B5 aux_loss_fn callback for iterate_dataset_ext. Lazily creates the
trainable audio_proj (Linear(rnn_size, C)) on first call, once the decoder
feature map's channel count C is known from its actual captured shape, and
registers it with the optimizer immediately (train_model.py's optimizer is
built from network.parameters() before iterate_dataset ever runs, so a
module added to `network` afterward needs this explicit registration or its
parameters would silently never receive a gradient step)."""
import torch.nn as nn

from extensions.hooks.position_decoder import center_of_mass_xy
from extensions.losses.dense_contrastive_aux import dense_contrastive_aux_loss

NUM_NEGATIVES = 32
EXCLUDE_RADIUS_PX = 30
TEMPERATURE = 0.07


def b5_aux_loss(pred, y_batch, rnn_out, decoder_feature, seq_len, bs, network, optimizer, pitch_batch=None):
    if decoder_feature is None:
        raise RuntimeError('B5 requires decoder_feature_stage set in iterate_dataset_ext')

    C = decoder_feature.shape[1]
    if not hasattr(network, '_ext_b5_audio_proj'):
        audio_proj = nn.Linear(rnn_out.shape[-1], C).to(decoder_feature.device)
        network.add_module('_ext_b5_audio_proj', audio_proj)
        optimizer.add_param_group({'params': audio_proj.parameters()})
    audio_proj = network._ext_b5_audio_proj

    H, W = pred.shape[-2], pred.shape[-1]
    y_grid = y_batch.view(seq_len, bs, H, W)
    gt_xy = center_of_mass_xy(y_grid).view(seq_len * bs, 2)   # (seq_len*bs, 2), matches decoder_feature's batch dim

    loss = dense_contrastive_aux_loss(
        decoder_feature, rnn_out, gt_xy, score_hw=(H, W), audio_proj=audio_proj,
        num_negatives=NUM_NEGATIVES, exclude_radius_px=EXCLUDE_RADIUS_PX, temperature=TEMPERATURE)
    return loss, {'dense_contrastive': loss.detach()}
