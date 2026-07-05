"""B4 aux_loss_fn callback for iterate_dataset_ext -- wraps
temporal_consistency_loss with the (pred, y_batch, seq_len, bs) reshape.

Positions are normalized to [0,1] by image (H,W) before the loss: raw pixel
coordinates (up to ~400px) make the L1 term ~100x larger than dice loss's
O(1) scale, so with aux_loss_weight=1.0 (matching CB_TA-Ext.md's spec) the
combined loss would be almost entirely the temporal term -- the exact
pixel/gradient-scale mismatch already diagnosed and fixed in the CADP work
this session (see mymodel/cadp/m0*_train.py PIX_SCALE comments)."""
from extensions.hooks.position_decoder import soft_argmax_xy, center_of_mass_xy
from extensions.losses.temporal_consistency import temporal_consistency_loss


def b4_aux_loss(pred, y_batch, rnn_out, decoder_feature, seq_len, bs, network=None, optimizer=None, pitch_batch=None):
    H, W = pred.shape[-2], pred.shape[-1]
    pred_grid = pred.view(seq_len, bs, H, W)
    y_grid = y_batch.view(seq_len, bs, H, W)

    pred_xy = soft_argmax_xy(pred_grid)     # (seq_len, bs, 2), pixel space
    gt_xy = center_of_mass_xy(y_grid)        # (seq_len, bs, 2), pixel space

    scale = pred_xy.new_tensor([max(W - 1, 1), max(H - 1, 1)])
    loss = temporal_consistency_loss(pred_xy / scale, gt_xy / scale)
    return loss, {'temporal_consistency': loss.detach()}
