#!/bin/bash
#SBATCH --job-name=cpjku-official-pretrained-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_pretrained_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_pretrained_cpu-%j.log

# CPU-routed duplicate of eval_cpjku_official_pretrained.sh (job 65455162),
# stuck PENDING/ReqNodeNotAvail on the a100-constrained GPU partition for a
# long stretch. eval_model.py already auto-falls-back to CPU
# (torch.device('cuda' if torch.cuda.is_available() else 'cpu')) and CB_TA
# is tiny (942K params) -- same CPU workaround already validated this
# session for F7/per-onset-diag. Separate OUT dir (cpjku_official_pretrained_cpu)
# to avoid colliding with the still-queued GPU duplicate's output if it ever
# also runs. Extended time budget (2h vs original 1h) since CPU inference on
# 94 pieces is slower than A100, though this is a single small model (not a
# 3-model ensemble), so should still be well under 2h.

echo "Job started on $(hostname) at $(date)"

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
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
OUT=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_pretrained_cpu
mkdir -p "$OUT"

cd "$REPO/audio_conditioned_unet"

echo "=== Eval TRUE official CB_TA pretrained weights (CPU): msmd_test, onset timing ==="
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
