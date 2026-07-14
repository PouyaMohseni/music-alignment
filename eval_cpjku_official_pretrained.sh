#!/bin/bash
#SBATCH --job-name=cpjku-official-pretrained
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_pretrained-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_pretrained-%j.log

# Table 3 (onset timing) eval against the TRUE official CB_TA pretrained
# weights bundled with the paper's own repo (third_party/cpjku_unet/models/CB_TA/best_model.pt)
# -- NOT one of our own trained checkpoints. eval_model.py was patched
# (d4e4c69) to also print per-piece mean/median onset error in seconds
# alongside the threshold percentages, so we can compare the actual error
# DISTRIBUTION per piece against our own F4 result, not just aggregate stats.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate

export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_pretrained
mkdir -p "$OUT"

cd "$REPO/audio_conditioned_unet"

echo "=== Eval TRUE official CB_TA pretrained weights: msmd_test, onset timing ==="
python eval_model.py \
    --param_path  "$REPO/models/CB_TA/best_model.pt" \
    --test_dir    ../data/msmd/msmd_test \
    --config      configs/msmd.yaml \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats \
    --dump_raw_onsets "$OUT/raw_onset_errors.json" \
    2>&1 | tee "$OUT/eval_onsets.log"

echo ""
echo "Job finished at $(date)"
