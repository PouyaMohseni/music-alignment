# Music Alignment Research Project

End-to-end audio-to-score alignment using multi-modal learning (MERT + ViT, SoftDTW).

## Structure
- `data/`     — datasets (MSMD, RWC, synthetic)
- `papers/`   — reproductions of baseline papers
- `mymodel/`  — custom model versions (v1–v3)
- `configs/`  — Hydra YAML configs
- `notebooks/`— EDA and visualizations
- `results/`  — logs, checkpoints, W&B exports
- `tests/`    — unit tests

## Papers Implemented
1. **Henkel et al. 2019** — Score Following as a Multi-Modal RL Problem (TISMIR)
2. **Dorfer et al. 2017** — End-to-End Cross-Modal Audio-Sheet Music Retrieval (ISMIR)

## My Model Versions
- `v1_baseline`  — MERT + ViT, late fusion, SoftDTW
- `v2_crossattn` — Cross-attention fusion, SoftDTW
- `v3_softdtw`   — Full model: cross-attention + SoftDTW + auxiliary InfoNCE
