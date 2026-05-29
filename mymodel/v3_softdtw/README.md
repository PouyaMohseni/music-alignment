# v3 — Full Proposed Model

- Audio encoder : MERT-v1-95M (LoRA)
- Score encoder : ViT-Base-patch16 (LoRA)
- Fusion        : cross-attention
- Loss          : SoftDTW (primary) + InfoNCE × 0.1 (auxiliary)
- Training      : end-to-end, sequential

This is the main proposed model. Adds temporal structure awareness to v2.
