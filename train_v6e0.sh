#!/bin/bash
#SBATCH --job-name=music-v6e0
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v6e0_pitch_aligned-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v6e0_pitch_aligned-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v6e0_pitch_aligned

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

V3_CKPT=$(ls results/v3_all/checkpoint_*.pt 2>/dev/null | sort | tail -1)
[ -z "$V3_CKPT" ] && echo "ERROR: no v3_all checkpoint" && exit 1
echo "Warm-starting from: $V3_CKPT"

# E0: v5i architecture + CORRECTLY-WIRED pitch supervision (pitch_on_aligned=true).
# The clean test of whether the pitch lever was real — controlled against v5i (2.38s).
python -m mymodel.v5_recurrent.train \
  --config configs/v6e0_pitch_aligned.yaml \
  init_v3_checkpoint=$V3_CKPT \
  data.emb_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/embeddings_all_tar \
  data.processed_root=data/MSMD/processed_all

echo "Job finished at $(date)"
