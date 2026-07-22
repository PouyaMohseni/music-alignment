"""M1 -- Monotonic cross-modal alignment for optical score-following.

Reformulates score-following from per-frame heatmap segmentation (Henkel /
CB_TA / every B-series variant) into differentiable monotonic sequence
alignment: an audio-frame x score-column alignment matrix supervised by a
forward-sum (CTC-style) monotonic-path likelihood, bootstrapped by a
beta-binomial near-diagonal prior, and decoded at inference with a monotonic
Viterbi pass -- so the global "position only moves forward" constraint is
enforced in both training and inference, rather than left to an LSTM to
implicitly learn. Directly targets the repeat-ambiguity error mode confirmed
as this project's dominant failure (per-onset diagnostic, task #24), which a
per-frame classifier structurally cannot resolve.

Grounding:
  - forward-sum / beta-binomial monotonic alignment: Badlani et al., "One TTS
    Alignment To Rule Them All" (2021/2022); RAD-TTS.
  - differentiable dynamic programming: Mensch & Blondel (2018); Cuturi &
    Blondel soft-DTW (2017) -- the same DP family already used in
    mymodel/d1_align_matrix/losses.py, whose numerics conventions
    (finite sentinel, logsumexp, NaN guards) this package matches.

Phase 0 (this commit): the four core algorithm pieces as standalone, GPU-free,
unit-tested modules (forward_sum, monotonic_decode, beta_binomial_prior,
repeat_unroll). No training wiring yet -- correctness of the DP, decode,
prior, and repeat-unrolling is proven against brute-force references before
any GPU time is spent. See scripts/smoke_test_monotonic_alignment.py.
"""
