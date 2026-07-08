#!/bin/bash
#SBATCH --job-name=mert-trainval-emb
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_trainval_emb-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_trainval_emb-%j.log

# D1 prerequisite: precompute MERT embeddings for the WHOLE-PIECE train+val
# performance wavs (cpjku_fmt/performance/<piece>.wav), matching the strip
# scores in cpjku_fmt/score. Test-split embeddings already exist under
# mert_emb_zenodo/cpjku_fmt_test_eval (built earlier for B1a eval); D1 reuses
# those for its own test eval, so this only fills train+val. Output goes to a
# shared dir keyed by whole-piece name so D1's dataset can look up
# <piece>.npy uniformly across splits.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

OUT=/scratch/pmohseni/mert_emb_zenodo/cpjku_fmt_wholepiece

for split in train val; do
    echo "=== $split ==="
    python scripts/precompute_mert_test_eval.py \
        --wav_dir    data/MSMD/cpjku_fmt/performance \
        --split_file data/MSMD/cpjku_fmt/split_${split}.yaml \
        --out_dir    "$OUT"
done

echo "Job finished at $(date)"
