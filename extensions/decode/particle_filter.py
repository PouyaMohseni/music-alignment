"""C3 -- Bayesian particle-filter decoder for score-position tracking at eval
time. No retraining: this only replaces the DECODE step (threshold +
center-of-mass -> particle filter) on top of an already-trained network's own
per-frame heatmap output, so it runs against CB_TA's bundled pretrained model
or any already-trained checkpoint in this project unchanged.

Score strips here are single-staff (per CLAUDE.md: "single-staff strips this
is always [[H//2], [0]]"), so tracking reduces to 1-D x-position -- rows are
marginalized out of the heatmap for the observation likelihood, matching
eval_official.py's own decode which only ever reads com_np[1] (the x
component of utils.center_of_mass's [y, x] output) for scoring.
"""
from __future__ import annotations
import numpy as np


def heatmap_to_x_marginal(heatmap_2d: np.ndarray) -> np.ndarray:
    """heatmap_2d: (H, W) raw sigmoid output. Returns (W,) column-sum marginal
    -- valid for single-staff strips where all vertical extent belongs to one
    staff, so marginalizing over rows loses no position information."""
    return heatmap_2d.sum(axis=0)


class ParticleFilterXTracker:
    """Tracks 1-D x-position across frames using the network's own per-frame
    heatmap as the observation likelihood.

    Motion model: constant-velocity + Gaussian process noise. Velocity is
    re-estimated online via an exponential moving average of the filter's own
    recent frame-to-frame displacement -- deliberately NOT a full
    tempo-tracking HMM; this is meant to be a light, robust prior that damps
    single-frame heatmap noise (e.g. a momentary bad observation shouldn't
    yank the estimate), not a sophisticated tempo model.
    """

    def __init__(self, n_particles: int = 200, process_noise_std: float = 3.0,
                 velocity_ema_alpha: float = 0.3, resample_frac: float = 0.5,
                 init_std: float = 5.0, seed: int = 0):
        self.n_particles = n_particles
        self.process_noise_std = process_noise_std
        self.velocity_ema_alpha = velocity_ema_alpha
        self.resample_threshold = resample_frac * n_particles
        self.init_std = init_std
        self.rng = np.random.default_rng(seed)

        self.particles = None      # (N,) x positions
        self.weights = None        # (N,) normalized weights
        self.velocity = 0.0        # running EMA of per-frame x displacement
        self.last_estimate = None

    def _init_particles(self, x0: float, w_max: float):
        self.particles = np.clip(
            self.rng.normal(x0, self.init_std, size=self.n_particles), 0, w_max)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def _effective_sample_size(self) -> float:
        return 1.0 / np.sum(self.weights ** 2)

    def _systematic_resample(self):
        N = self.n_particles
        positions = (self.rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        idx = np.searchsorted(cumsum, positions)
        self.particles = self.particles[idx]
        self.weights = np.full(N, 1.0 / N)

    def step(self, x_marginal: np.ndarray) -> float:
        """x_marginal: (W,) non-negative likelihood over x bins (one bin per
        score-pixel column). Returns the weighted-mean x-position estimate
        for this frame. Call once per frame, in temporal order, per piece
        (construct a fresh tracker per piece -- state is not reset otherwise)."""
        W = x_marginal.shape[0]

        if self.particles is None:
            x0 = float(np.argmax(x_marginal)) if x_marginal.sum() > 0 else W / 2.0
            self._init_particles(x0, W - 1)
            self.last_estimate = x0
            return x0

        # --- motion update ---
        noise = self.rng.normal(0.0, self.process_noise_std, size=self.n_particles)
        self.particles = np.clip(self.particles + self.velocity + noise, 0, W - 1)

        # --- observation update ---
        idx = np.clip(np.round(self.particles).astype(int), 0, W - 1)
        likelihood = x_marginal[idx] + 1e-6
        self.weights = self.weights * likelihood
        wsum = self.weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            # degenerate frame (e.g. all-zero heatmap) -- fall back to uniform
            # rather than dividing by zero/propagating NaN weights forward.
            self.weights = np.full(self.n_particles, 1.0 / self.n_particles)
        else:
            self.weights = self.weights / wsum

        if self._effective_sample_size() < self.resample_threshold:
            self._systematic_resample()

        estimate = float(np.sum(self.particles * self.weights))

        if self.last_estimate is not None:
            delta = estimate - self.last_estimate
            self.velocity = (self.velocity_ema_alpha * delta
                              + (1 - self.velocity_ema_alpha) * self.velocity)
        self.last_estimate = estimate

        return estimate
