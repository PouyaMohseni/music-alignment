"""B3 aux_loss_fn callback for iterate_dataset_ext. Lazily creates
LocalINRRefiner on first call (needs decoder_feature's actual channel count)
and registers it with the optimizer -- same reason as B5's audio_proj:
train_model.py builds the optimizer before iterate_dataset ever runs."""
from extensions.hooks.position_decoder import thresholded_center_of_mass_xy, center_of_mass_xy
from extensions.heads.inr_subpixel_head import LocalINRRefiner, make_query_grid, heatmap_inr_loss_2d

WINDOW_PX = 8
QUERY_RESOLUTION_MULTIPLIER = 4
HEATMAP_SIGMA_PX = 5.0


def b3_aux_loss(pred, y_batch, rnn_out, decoder_feature, seq_len, bs, network, optimizer):
    if decoder_feature is None:
        raise RuntimeError('B3 requires decoder_feature_stage set in iterate_dataset_ext')

    C = decoder_feature.shape[1]
    if not hasattr(network, '_ext_b3_inr_refiner'):
        refiner = LocalINRRefiner(C, window_px=WINDOW_PX).to(decoder_feature.device)
        network.add_module('_ext_b3_inr_refiner', refiner)
        optimizer.add_param_group({'params': refiner.parameters()})
    refiner = network._ext_b3_inr_refiner

    H, W = pred.shape[-2], pred.shape[-1]
    pred_grid = pred.view(seq_len * bs, H, W)
    y_grid = y_batch.view(seq_len * bs, H, W)

    coarse_peak = thresholded_center_of_mass_xy(pred_grid)   # (seq_len*bs, 2), pixel space
    gt_xy = center_of_mass_xy(y_grid)                          # (seq_len*bs, 2), pixel space
    gt_offset = gt_xy - coarse_peak

    query_offsets = make_query_grid(WINDOW_PX, QUERY_RESOLUTION_MULTIPLIER, decoder_feature.device)
    _, confidence = refiner(decoder_feature, coarse_peak, query_offsets, score_hw=(H, W))

    loss = heatmap_inr_loss_2d(confidence, query_offsets, gt_offset, sigma_px=HEATMAP_SIGMA_PX)
    return loss, {'inr_subpixel': loss.detach()}
