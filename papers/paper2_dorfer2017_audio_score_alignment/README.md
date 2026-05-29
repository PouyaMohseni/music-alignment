# Paper 2 — Dorfer et al. 2017

**Title:** End-to-End Cross-Modal Audio-Sheet Music Retrieval
**Venue:** ISMIR 2017
**Authors:** Matthias Dorfer, Andreas Arzt, Gerhard Widmer

## Summary
Learns a shared embedding space between audio spectrograms and sheet music
image patches via a siamese network with a pairwise ranking loss. Retrieval
is done by nearest-neighbour search in the embedding space.

## Key Components
- Audio branch: CNN on log-CQT spectrogram (92 bins)
- Score branch: CNN on sheet music image patches
- Loss: pairwise ranking (matched vs unmatched pairs)
- No sequential model — each snippet is independent

## Reproduce Checklist
- [ ] Audio CNN (log-CQT, 92 bins x ~43 frames)
- [ ] Score CNN (image patch)
- [ ] Pairwise ranking loss
- [ ] MSMD evaluation: Recall@1, Recall@10

## Deviations from Paper
*(fill in as you implement)*

## Reported Results
- Dataset: MSMD
- Metric: Recall@k (retrieval accuracy)
- Snippet: 1 second (~43 frames @ 43 fps)
- No repeats handling (snippets treated independently)
