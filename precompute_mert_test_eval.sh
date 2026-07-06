#!/bin/bash
#SBATCH --job-name=mert-test-eval-emb
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_test_eval_emb-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_test_eval_emb-%j.log

# Precompute MERT embeddings for B1a's eval -- see
# scripts/precompute_mert_test_eval.py for why this is a separate,
# eval-only set of embeddings from the ones used to train B1a.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

python scripts/precompute_mert_test_eval.py \
    --wav_dir    data/MSMD/cpjku_fmt/performance \
    --split_file data/MSMD/cpjku_fmt/split_test.yaml \
    --out_dir    /scratch/pmohseni/mert_emb_zenodo/cpjku_fmt_test_eval

echo "Job finished at $(date)"
