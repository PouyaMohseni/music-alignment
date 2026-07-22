"""Phase 0 smoke test for the M1 monotonic-alignment package
(extensions/alignment/). GPU-free. Every algorithm piece is checked against
an INDEPENDENT reference (brute-force path enumeration / analytic property),
not just "runs without error" -- the same rigor that caught the v12 DTW
backtrack bug and the gated-FiLM init-clobber bug.

Checks:
  forward_sum   -- DP logZ vs brute-force enumeration of every monotonic path;
                   >= Viterbi score; gradcheck (double precision) of the DP
                   backward pass; diagonal beats anti-diagonal; batched==loop.
  viterbi       -- score & path vs brute-force argmax path; monotone, endpoints,
                   step set {0,1}; repeat resolved by global context.
  beta_binomial -- rows are distributions; per-row mode monotone & diagonal.
  repeat_unroll -- exact unrolled sequence; monotone-virtual -> sawtooth printed
                   with a genuine backward jump; posterior fold; x-map.
"""
import sys
from itertools import combinations

sys.path.insert(0, '/lustre06/project/6002780/pmohseni/music-alignment')

import numpy as np
import torch

from extensions.alignment.forward_sum import (
    forward_sum_logZ, forward_sum_loss, forward_sum_loss_batched, _NEG)
from extensions.alignment.monotonic_decode import (
    viterbi_path, expected_position, path_to_position)
from extensions.alignment.beta_binomial_prior import beta_binomial_prior, beta_binomial_log_prior
from extensions.alignment.repeat_unroll import (
    unroll_repeats, printed_path_from_virtual, fold_posterior_to_printed, unroll_column_x)


def _all_monotone_paths(T, N):
    """Every stay-or-advance-by-one path from (0,0) to (T-1,N-1): choose which
    (N-1) of the (T-1) inter-frame steps are advances."""
    for adv_steps in combinations(range(1, T), N - 1):
        path = np.zeros(T, dtype=np.int64)
        col = 0
        adv = set(adv_steps)
        for t in range(1, T):
            if t in adv:
                col += 1
            path[t] = col
        yield path


def _brute_logZ(log_emit):
    le = log_emit.detach().numpy()
    T, N = le.shape
    logsums = [le[np.arange(T), p].sum() for p in _all_monotone_paths(T, N)]
    m = max(logsums)
    return m + np.log(sum(np.exp(s - m) for s in logsums))


def _brute_best(log_emit):
    le = log_emit.detach().numpy()
    T, N = le.shape
    best_s, best_p = -np.inf, None
    for p in _all_monotone_paths(T, N):
        s = le[np.arange(T), p].sum()
        if s > best_s:
            best_s, best_p = s, p
    return best_s, best_p


print("=== forward_sum: DP logZ vs brute-force path enumeration ===")
torch.manual_seed(0)
for (T, N) in [(7, 4), (6, 3), (10, 5), (5, 5)]:
    le = torch.randn(T, N, dtype=torch.float64)
    dp = forward_sum_logZ(le).item()
    bf = _brute_logZ(le)
    assert abs(dp - bf) < 1e-6, f"FAIL logZ T={T} N={N}: dp={dp:.8f} brute={bf:.8f}"
    print(f"  T={T:2d} N={N}: dp logZ={dp:+.6f}  brute={bf:+.6f}  |diff|={abs(dp-bf):.2e}  PASS")

print()
print("=== forward_sum: gradcheck (double precision) of the DP backward pass ===")
le = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
ok = torch.autograd.gradcheck(forward_sum_logZ, (le,), eps=1e-6, atol=1e-4, rtol=1e-3)
print(f"  torch.autograd.gradcheck(forward_sum_logZ): {ok}  {'PASS' if ok else 'FAIL'}")
assert ok, "FAIL: forward_sum_logZ gradient does not match finite differences"

print()
print("=== forward_sum: logZ >= Viterbi score (logsumexp >= max) ===")
le = torch.randn(9, 4, dtype=torch.float64)
lz = forward_sum_logZ(le).item()
_, vscore = viterbi_path(le)
assert lz >= vscore.item() - 1e-9, f"FAIL: logZ {lz} < viterbi {vscore.item()}"
print(f"  logZ={lz:+.5f} >= viterbi={vscore.item():+.5f}  PASS")

print()
print("=== forward_sum_loss: diagonal alignment beats anti-diagonal ===")
T, N = 12, 6
diag = torch.full((T, N), -3.0)
for t in range(T):
    diag[t, min(int(t * N / T), N - 1)] = 5.0
anti = torch.flip(diag, dims=[1])
ld, la = forward_sum_loss(diag).item(), forward_sum_loss(anti).item()
assert ld < la, f"FAIL: diagonal loss {ld} !< anti-diagonal loss {la}"
print(f"  diagonal loss={ld:.4f} < anti-diagonal loss={la:.4f}  PASS")

print()
print("=== forward_sum_loss_batched == per-item loop ===")
scores = torch.randn(3, 12, 6)
flens = torch.tensor([12, 10, 11])
clens = torch.tensor([6, 5, 4])
manual = torch.stack([forward_sum_loss(scores[b, :int(flens[b]), :int(clens[b])])
                      for b in range(3)]).mean()
batched = forward_sum_loss_batched(scores, flens, clens)
assert torch.allclose(manual, batched, atol=1e-6), f"FAIL: {manual} vs {batched}"
print(f"  loop={manual.item():.6f}  batched={batched.item():.6f}  PASS")


print()
print("=== viterbi: score & path vs brute-force argmax ===")
for (T, N) in [(7, 4), (9, 3), (11, 6)]:
    le = torch.randn(T, N, dtype=torch.float64)
    path, score = viterbi_path(le)
    bf_score, bf_path = _brute_best(le)
    assert abs(score.item() - bf_score) < 1e-9, f"FAIL viterbi score T={T} N={N}"
    assert np.array_equal(path.numpy(), bf_path), f"FAIL viterbi path T={T} N={N}: {path.numpy()} vs {bf_path}"
    diffs = np.diff(path.numpy())
    assert path[0] == 0 and path[-1] == N - 1, "FAIL: endpoints"
    assert set(np.unique(diffs).tolist()) <= {0, 1}, f"FAIL: step set {np.unique(diffs)}"
    print(f"  T={T:2d} N={N}: score match, path match, monotone step-set OK  PASS")

print()
print("=== viterbi: a repeat is resolved by GLOBAL context (the whole point) ===")
# 8 frames, 3 columns. Columns 0 and 2 look IDENTICAL locally (same emission
# profile); a per-frame argmax on the last frame would tie 0 vs 2. The monotone
# path forces the late frames onto column 2 because it has already advanced.
T, N = 8, 3
le = torch.full((T, N), -5.0)
le[0:3, 0] = 2.0                       # early frames clearly column 0
le[3:5, 1] = 2.0                       # middle frames column 1
le[5:8, 0] = 2.0; le[5:8, 2] = 2.0     # late frames AMBIGUOUS between col 0 and col 2
path, _ = viterbi_path(le)
per_frame_argmax = le.argmax(dim=1)
assert path[-1].item() == 2, f"FAIL: monotone decode did not resolve repeat to col 2: {path.tolist()}"
assert per_frame_argmax[-1].item() in (0, 2), "sanity: per-frame is genuinely ambiguous"
print(f"  per-frame argmax (last 3 frames)={per_frame_argmax[5:].tolist()} (ambiguous, may pick 0)")
print(f"  monotone viterbi path={path.tolist()} -> last frame correctly col 2  PASS")

print()
print("=== expected_position / path_to_position readouts ===")
col_x = torch.tensor([10.0, 20.0, 30.0])
probs = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.5, 0.5]])
ep = expected_position(probs, col_x, apply_softmax=False)
assert torch.allclose(ep, torch.tensor([11.0, 25.0]), atol=1e-5), f"FAIL expected_position {ep}"
pp = path_to_position(torch.tensor([0, 1, 2]), col_x)
assert torch.allclose(pp, col_x), f"FAIL path_to_position {pp}"
print(f"  expected_position={ep.tolist()} (want [11.0, 25.0])  path_to_position OK  PASS")


print()
print("=== beta_binomial_prior: distribution rows + monotone diagonal mode ===")
P = beta_binomial_prior(20, 8, scale=1.0)
assert P.shape == (20, 8), f"FAIL shape {P.shape}"
assert np.allclose(P.sum(axis=1), 1.0, atol=1e-4), "FAIL: rows not distributions"
modes = P.argmax(axis=1)
assert np.all(np.diff(modes) >= 0), f"FAIL: per-row mode not monotone: {modes}"
assert modes[0] == 0 and modes[-1] == 7, f"FAIL: corners not diagonal: {modes[0]}, {modes[-1]}"
lp = beta_binomial_log_prior(20, 8)
assert lp.shape == (20, 8) and torch.isfinite(lp).all(), "FAIL: log-prior not finite"
print(f"  shape OK, rows sum to 1, per-row mode monotone {modes.tolist()} (0..7)  PASS")


print()
print("=== repeat_unroll: exact sequence + monotone-virtual -> sawtooth printed ===")
v2p = unroll_repeats(10, [(3, 6)])
expected = np.array([0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 9])
assert np.array_equal(v2p, expected), f"FAIL unroll: {v2p.tolist()} vs {expected.tolist()}"
# a monotone path in VIRTUAL space (visit every virtual column once):
virtual_path = np.arange(len(v2p))
printed = printed_path_from_virtual(virtual_path, v2p)
assert np.array_equal(printed, expected), "FAIL: printed path fold"
# the printed trajectory is NON-monotone (has the backward jump 6 -> 3):
assert (np.diff(printed) < 0).any(), "FAIL: expected a backward jump in printed space"
jump_idx = int(np.argmin(np.diff(printed)))
assert printed[jump_idx] == 6 and printed[jump_idx + 1] == 3, "FAIL: jump not 6->3"
print(f"  virtual_to_printed={v2p.tolist()}")
print(f"  monotone virtual path folds to printed sawtooth with 6->3 backward jump  PASS")

print()
print("=== repeat_unroll: posterior fold + column-x map ===")
V = len(v2p)
post = np.zeros((2, V), dtype=np.float32)
post[0, 3] = 0.6    # virtual col 3 -> printed 3
post[0, 7] = 0.4    # virtual col 7 -> printed 3 (the repeat) -> should sum onto printed 3
folded = fold_posterior_to_printed(post, v2p, 10)
assert abs(folded[0, 3] - 1.0) < 1e-6, f"FAIL: fold did not sum repeat mass onto printed col 3: {folded[0,3]}"
col_x = np.arange(10, dtype=np.float32) * 5.0
vx = unroll_column_x(col_x, v2p)
assert vx[7] == col_x[3], "FAIL: unroll_column_x"
print(f"  two virtual cols (3 and 7) rendering printed col 3 summed to {folded[0,3]:.3f}  PASS")

print()
print("ALL PHASE-0 MONOTONIC-ALIGNMENT TESTS PASSED")
