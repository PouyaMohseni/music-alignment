#!/bin/bash
#SBATCH --job-name=cpjku-eval-cpu
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_cpu-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/cpjku_official_cpu-%j.log

# CPU-routed duplicate of eval_cpjku_official.sh (job 65667432), which has
# been stuck PENDING/ReqNodeNotAvail on the saturated GPU partition. CB_TA
# is a tiny (942K param) model and eval_official.py already auto-falls-back
# to CPU (torch.device('cuda' if torch.cuda.is_available() else 'cpu')), so
# dropping --gres=gpu:1 is the only change needed -- same workaround already
# validated this session for F7 and per-onset-diag. This is the confirmation
# run for whether the real-madmom spectrogram cache (commit 24cbaa0) closes
# the harness gap back toward the paper's ~85%.

echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt
mkdir -p results

echo "=== Step 1: Convert MSMD processed -> CPJKU format ==="
python -m mymodel.cpjku_adapter.convert \
    --processed $PROC \
    --out       $CPJKU_FMT \
    --splits    test val train

echo "=== Step 2: Eval with CPJKU pre-trained CB_TA model (CPU, real-madmom spec cache) ==="
python -m mymodel.cpjku_adapter.eval_official \
    --cpjku_root  third_party/cpjku_unet \
    --cpjku_data  $CPJKU_FMT \
    --processed   $PROC \
    --model       CB_TA \
    --split       test \
    --batch_size  1 \
    --seq_len     8 \
    --scale_factor 3

echo "Job finished at $(date)"
