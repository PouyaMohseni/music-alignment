# Paper 1 — Henkel et al. 2019

**Title:** Score Following as a Multi-Modal Reinforcement Learning Problem
**Venue:** TISMIR 2(1), 2019
**Authors:** Florian Henkel, Stefan Balke, Matthias Dorfer, Gerhard Widmer

## Summary
Frames score following as an RL problem. An agent observes audio snippets +
score image patches and learns a policy to track position in the score in real time.

## Key Components
- State: (log-filterbank audio snippet, score image patch)
- Action: {stay, advance} in the score
- Reward: based on alignment accuracy
- Network: CNN encoders for both modalities + policy head (REINFORCE)

## Reproduce Checklist
- [ ] CNN audio encoder (log-filterbank input, ~92 bins)
- [ ] CNN score image encoder
- [ ] RL training loop (policy gradient / REINFORCE)
- [ ] Evaluation on MSMD test split

## Deviations from Paper
*(fill in as you implement)*

## Reported Results
- Dataset: MSMD (piano, synthetic rendered)
- Metric: tracking error (mean beat offset), % frames within threshold
- Snippet: ~1 second audio / matching score patch
