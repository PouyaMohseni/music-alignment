"""B2 -- Pitch Auxiliary, Correctly Wired.

Documented bug in prior work (REDESIGN.md 9.1): the pitch head read the
frozen precomputed embedding, not the FiLM-modulated feature the network
actually uses to localize -- gradient never reached anything load-bearing.
This attaches to the POST-FiLM decoder feature map (gradient-connected) and
the LSTM hidden state (the actual FiLM-conditioning vector), not any
frozen/detached input. See b2_callback.py's gradient-flow assertion, which
this exact bug motivates checking explicitly rather than trusting the loss
value alone.
"""
import torch.nn as nn

from extensions.hooks.film_feature_extractor import bilinear_sample, pixel_to_norm


class PitchAuxHead(nn.Module):
    def __init__(self, rnn_hidden, feature_channels, num_pitches=88):
        super().__init__()
        self.audio_pitch_head = nn.Linear(rnn_hidden, num_pitches)
        self.score_pitch_head = nn.Linear(feature_channels, num_pitches)

    def forward(self, rnn_out, film_decoder_feature_map, gt_xy_px, score_hw):
        """rnn_out: (B, rnn_hidden). film_decoder_feature_map: (B, C, H, W),
        gradient-connected. gt_xy_px: (B, 2) [x,y] in ORIGINAL score-pixel
        space. score_hw: (H_s, W_s) of that space."""
        audio_pitch_logits = self.audio_pitch_head(rnn_out)                    # (B, 88)

        gt_norm = pixel_to_norm(gt_xy_px, score_hw)
        sampled_feature = bilinear_sample(film_decoder_feature_map, gt_norm)     # (B, C)
        score_pitch_logits = self.score_pitch_head(sampled_feature)              # (B, 88)

        return audio_pitch_logits, score_pitch_logits, sampled_feature
