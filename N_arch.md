# N-track: temporal architectures for the teleport failure (2026-07-31)

## Diagnosis this track exists to fix

The 2026-07-31 eval sweep made the residual error legible. In the best model
(B1a native, **89.2%** pct@0.5s) the failing pieces do **not** degrade
gracefully:

| piece | median err | mean err |
|---|---|---|
| SchumannR op68-01-melodie | **0.000s** | 5.66s |
| SchumannR op68-06 | **0.000s** | 5.97s |
| SatieE gymnopedie-3 | **0.000s** | 9.46s |
| ChopinFF Op28-9 | 3.35s | 12.39s |

Median ~0 with a large mean = exact most of the time, catastrophically wrong
in **bursts**. Meanwhile ~75% of test pieces score 98-100%. So the remaining
headroom is **discrete global mislocalisation** (teleporting to a visually
similar passage, i.e. a repeat, dwelling, recovering) -- NOT precision.

Confirms the earlier per-onset diagnostic (task #24, repeat ambiguity).

## Constraints taken from measured results

| Evidence | Constraint |
|---|---|
| V-DINOv2 full visual-encoder swap = 6.9-9.2% | do not touch the conv visual encoder (resolution loss) |
| spatial FiLM 44.3%, cross-attn FiLM 71.1%, gated FiLM 82.9% vs plain FiLM 89.2% | do not REPLACE FiLM |
| B3 (+aux loss, warm-started on converged B1a) = **89.8%**, the only thing to beat B1a | additive-on-converged is the reliable pattern |

## The three pathways

All keep the visual encoder + FiLM intact and thread state through CPJKU's
existing 2-tuple `hidden`, so **iterate_dataset and eval_model.py are
unmodified** (they only zero/slice dim 1, which is exactly per-piece reset
and slot-drop).

- **N1** `extensions/heads/long_context_temporal.py` -- REPLACE the 1-layer
  LSTM with a two-tier memory Transformer (64 fine frames + 192 compressed
  slots x16 ~= 2.5 min of history). Rationale: "have I played this already?"
  needs the distant past, which a fixed-size recurrent state cannot hold.
- **N2** `extensions/heads/gated_memory_retrieval.py` -- KEEP the LSTM
  verbatim, add a silenced-at-init **lag-aware** retrieval read over the
  piece's own compressed history. The lag embedding is the point: a strong
  match to audio ~40s old is evidence of being on the SECOND pass, so the
  model can learn "advance", not "jump back".
- **N3** `extensions/heads/belief_propagation.py` -- KEEP the LSTM verbatim,
  add a silenced-at-init differentiable **Bayes filter** as a log-prior on the
  output heatmap: learned 2D transition kernel (general, so it can express a
  staff-system wrap, not just a forward step) + learned **uniform escape
  floor** so a wrong commitment stays recoverable rather than becoming
  lock-in.

Shared patch `extensions/hooks/temporal_arch_patch.py`;
warm-start loader with scoped allowlist `extensions/hooks/warm_start_load.py`.

## Validated before any GPU time (job 66849538, all passed)

- N2/N3 compute **exactly** stock CB_TA at init (`max|diff| = 0.00e+00`), so
  warm-starting from B1a is lossless and they start at 89.2%.
- Hidden contract, multi-chunk carry, per-slot reset, slot drop.
- Branches **unblock**: all new tensors receive gradient after one step.

### Bugs the smoke test caught (would each have silently wasted a 24h run)

1. **Zero multiplicative gate freezes its own branch.** `out = x + gate*f(x)`
   with `gate=0` gives `dL/dtheta_f = 0`. N3 self-recovers (its bias is a
   structured prior, so the gate has informative signal), but N2's branch
   output at init is random attention noise -- the gate would random-walk near
   zero and retrieval might never switch on, reporting ~89.2% and looking like
   "retrieval doesn't help". Fixed by silencing via a **zero-initialised final
   layer** instead (GPT-2 / ControlNet zero-conv): identity is still exact but
   `dL/d(out_proj.W) != 0`, so it moves on step one and unblocks the rest.
2. **`self.apply(initialize_weights)` clobbers the zero-init** -- orthogonal-
   inits every nn.Linear after construction. Same bug previously caught for
   GatedFiLM. Fixed with a `_zero_init` tag honoured by a wrapped
   initialize_weights.
3. **`super(ConditionalUNet, self)` global-name trap** (test-only): once the
   patch owns the module global, the original class cannot instantiate itself.

## Status

| exp | train job | eval |
|---|---|---|
| N1 long-context | 66850373 | `eval_n1_long_context_cpu.sh` |
| N2 memory retrieval | 66850394 | `eval_n2_memory_retrieval_cpu.sh` |
| N3 belief propagation | 66850395 | `eval_n3_belief_propagation_cpu.sh` |

All warm-start from B1a's `best_model.pt` on their first run and resume from
their own `latest_model.pt` afterwards. Eval runs on the CPU queue: unlike
MERT+DINOv2-crossattn (killed at 11/125 pieces), these modules sit in the
per-frame conditioning path, not inside the per-decoder-block loop.
