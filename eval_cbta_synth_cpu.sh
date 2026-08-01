#!/bin/bash
#SBATCH --job-name=eval-cbta-synth
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_cbta_synth-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_cbta_synth-%j.log
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1; export OPENBLAS_NUM_THREADS=1; export MKL_NUM_THREADS=1
cd third_party/cpjku_unet/audio_conditioned_unet
echo "=== paper's CB_TA on SYNTHETIC msmd_test (reference for the real-audio drop) ==="
python eval_model.py \
  --param_path ../models/CB_TA/best_model.pt \
  --test_dir ../data/msmd/msmd_test \
  --config configs/msmd.yaml \
  --scale_factor 3 --batch_size 1 --seq_len 128 --eval_onsets --piecewise_stats
echo "Job finished at $(date)"
