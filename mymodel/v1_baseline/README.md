# v1 — Baseline: MERT + ViT, Late Fusion

- Audio encoder : MERT-v1-95M (LoRA fine-tuned, rank=8)
- Score encoder : ViT-Base-patch16 (LoRA fine-tuned, rank=8)
- Fusion        : concatenate embeddings → 2-layer MLP
- Loss          : SoftDTW
- Training      : end-to-end

Ablation role: establishes foundation model encoder contribution vs CNN baselines.
