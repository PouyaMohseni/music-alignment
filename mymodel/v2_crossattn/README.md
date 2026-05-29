# v2 — Cross-Attention Fusion

- Audio encoder : MERT-v1-95M (LoRA)
- Score encoder : ViT-Base-patch16 (LoRA)
- Fusion        : cross-attention (audio queries ↔ score keys/values + vice versa)
- Loss          : SoftDTW
- Training      : end-to-end

Ablation role: tests cross-modal attention vs late fusion (v1).
