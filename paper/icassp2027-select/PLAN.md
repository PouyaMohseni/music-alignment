# ICASSP 2027 paper plan

Deadline: 16 Sep 2026 (full paper). Format: 4 pages of content + 1 page for
references / acknowledgements / ethics statement only. Not double-blind (names
on the PDF). Track: Audio and Acoustic Signal Processing, music signal
processing. Official kit in `kit/` (spconf.sty, IEEEbib.bst, Template.tex).

## What makes an ICASSP paper get in

Reviewers read the abstract, Fig. 1 and the main table first. We are accepted
if those three say, without the body: (1) there is a real problem, (2) we found
out *why* it happens with a clean measurement, (3) a simple fix built on that
finding wins clearly on a real benchmark, (4) the gain is not a fluke. So:

- one idea, stated in the title, repeated in abstract, intro, method, results
- a measurement that motivates the method (the candidate oracle), not a
  method in search of a motivation
- a main table against the published state of the art on the *same* real
  recordings, all five thresholds, so nobody has to trust our reproduction
- statistics a reviewer cannot argue with: piece-clustered bootstrap CIs,
  seed means, model selection that never touched the test set
- small and cheap: frozen detector, 23k parameters, no layout annotations
- honest limits in one short paragraph (it pre-empts the reviewer)

## The narrative

**Selection, not perception.** A state-of-the-art sheet-image tracker loses
20 points on real piano recordings. We show the loss is not in what the
detector *sees* but in how one position is *chosen*: on the real room-mic
recordings a correct candidate is among the detector's own top 256 at every
onset (offline oracle 100.0%), and even a causal oracle picking among them
scores 99.6%, while its argmax scores 80.0%. So we keep the detector frozen and
learn to select. A zero-parameter transition prior plus a 23k-parameter set
scorer reach 94.9% (seed mean 92.7), above a recent end-to-end system that
retrains the whole model and needs layout annotations at test time (88.3).
Analysis then explains what is left: two thirds is the tracker inheriting its
own mistakes, and the features that reduce it are exactly those that do not
depend on the tracker's history.

Everything in the paper serves that story. Things that do NOT go in because
they do not serve it: the twelve failed decoder mechanisms (one sentence at
most), pitch heads, tempo tracking, DAgger, the MERT/DINOv2 swap *unless* it
shows the decoder gain transfers across encoders (then it is a generality
result and goes in as a small table).

## Title

SELECTION, NOT PERCEPTION: LEARNING TO CHOOSE DETECTOR CANDIDATES FOR ROBUST
SHEET-IMAGE SCORE FOLLOWING

## Section and paragraph plan (target lengths in columns; 8 columns total)

**Abstract** (~150 words): problem, diagnosis numbers (100 / 99.6 / 80.0),
method in one sentence, headline 80.0 -> 94.9 and the comparison to CODA,
held-out robustness, propagation finding.

**1. Introduction** (~1.1 col)
- P1 task and why sheet images: page turning, accompaniment; most music exists
  only as printed scores; OMR is error-prone; deep models go straight from
  audio + page image to position.
- P2 the gap: trained on synthesized audio; published real-recording numbers
  drop (CYOLO-SB .893 -> .799; CUNet .855 -> .224). Existing remedies:
  augmentation, proprietary extra data, new end-to-end architectures with
  layout supervision (CODA).
- P3 our measurement: candidate oracles on the real recordings. The detector
  proposes the answer; argmax loses it. -> reframe robustness as selection.
- P4 contributions, three bullets: diagnosis; decoder (state of the art on the
  real recordings with a frozen detector, no layout annotations, 1.6% extra
  parameters); analysis of what remains (propagation, history-independent
  features).

**2. Related work** (~0.45 col): symbolic score following (Dannenberg, Cont,
Dixon, Raphael, Nakamura); image-based (Dorfer, Henkel RL, CUNet, CYOLO(-SB),
CODA); re-scoring detections (learned NMS, relation networks); exposure bias
(scheduled sampling, DAgger). One sentence each group, positioned against us.

**3. Method** (~1.6 col) with Fig. 1 (qualitative example)
- 3.1 Candidates from a frozen detector: notation, K = 256 of 8,112 boxes,
  unrolled coordinate u, argmax baseline. Eq. (1) argmax.
- 3.2 Transition prior: Eq. (2) floored Gaussian on displacement with
  time-scaled mean. Zero parameters.
- 3.3 Candidate scorer: feature groups (confidence, geometry, motion w.r.t.
  history, detector P3 feature, neighbourhood), Deep-Sets architecture,
  Eq. (3)-(4). Parameter count.
- 3.4 Training: listwise soft cross-entropy Eq. (5), teacher-forced history
  with heavy-tailed perturbation, checkpoint by validation rollout.
- 3.5 Decoding: Eq. (6) blend, causal greedy.

**4. Experiments** (~2.6 col) with Table 1 (full width), Table 2, Fig. 2
- 4.1 Setup: MSMD train/valid; real recordings (Setting II, 16 pieces, 25
  pages, 4,149 onsets, room mic, Yamaha AvantGrand); held-out 80 pieces with
  real IR + noise at 4 SNRs; metric; bootstrap; selection protocol.
- 4.2 Main result (Table 1): published systems from [CODA] + our reproduction
  + decoder rows; every-frame row. Text: all thresholds, CI.
- 4.3 Ablation (Table 2): the decoder ladder, seed means, held-out column,
  parameters; (optional) encoder swap.
- 4.4 Robustness to noise (Fig. 2b).
- 4.5 What is left (Fig. 2a): teacher-forced decomposition, failure types.

**5. Conclusion and limitations** (~0.35 col): summary; limits (correct page
given, piano only, repeats untested, 16 test pieces); future: other
history-independent evidence (pitch).

**Page 5**: references (~24), ethics statement (public data only).

## Open items (filled when jobs finish)

- every-frame tracking numbers (job 2882923): the fair comparison with CODA
- encoder swap cells (jobs 2884188-2884199): include only if it shows transfer
- qualitative figure frame choice
