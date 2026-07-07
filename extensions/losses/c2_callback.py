"""C2 aux_loss_fn callback for iterate_dataset_ext -- wraps soft_dtw_loss
with the (pred, y_batch, seq_len, bs) reshape, matching B4's b4_callback.py
exactly (same [0,1] position normalization discipline -- see that file's
comment for why: raw pixel coords make the loss ~100x dice's O(1) scale
otherwise)."""
from extensions.hooks.position_decoder import soft_argmax_xy, center_of_mass_xy
from extensions.losses.soft_dtw import soft_dtw_loss


def c2_aux_loss(pred, y_batch, rnn_out, decoder_feature, seq_len, bs, network=None, optimizer=None, pitch_batch=None):
    H, W = pred.shape[-2], pred.shape[-1]
    pred_grid = pred.view(seq_len, bs, H, W)
    y_grid = y_batch.view(seq_len, bs, H, W)

    pred_xy = soft_argmax_xy(pred_grid)     # (seq_len, bs, 2), pixel space
    gt_xy = center_of_mass_xy(y_grid)        # (seq_len, bs, 2), pixel space

    scale = pred_xy.new_tensor([max(W - 1, 1), max(H - 1, 1)])
    loss = soft_dtw_loss(pred_xy / scale, gt_xy / scale)
    return loss, {'soft_dtw': loss.detach()}
