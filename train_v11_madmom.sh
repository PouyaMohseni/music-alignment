#!/bin/bash
#SBATCH --job-name=v11-madmom
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v11_madmom-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v11_madmom-%j.log

# Fresh v11 run: same CB_TA-faithful architecture, but real-madmom spectrogram
# input (data/MSMD/cpjku_fmt/spec_madmom/, precomputed by
# mymodel.cpjku_adapter.precompute_madmom_specs under venv_cpjku310) instead
# of the mel-spectrogram approximation that the original v11 run used.
#
# v11 (mel-spec) plateaued at 56.2%/58.4% pct@0.5s -- below even our own
# MERT ensemble (73.6%), let alone the paper's 85.1%, despite matching their
# architecture exactly. This tests whether that gap was a spectrogram
# feature-fidelity problem rather than an architecture/capacity one. No
# warm-start from the old checkpoint: the input feature distribution changed
# entirely, so the CBEncoder's learned first layers need to relearn from
# scratch regardless.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v11_madmom

module load gcc opencv
source .venv/bin/activate

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed
CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt

echo "=== Train v11-madmom: CPJKU full-strip BPTT, real-madmom spectrogram ==="
RESUME_FLAG=""
if ls results/v11_madmom/checkpoint_epoch*.pt 1>/dev/null 2>&1; then
    echo "Found existing checkpoint — resuming."
    RESUME_FLAG="--resume"
fi
python -m mymodel.v11_cpjku_fullstrip.train \
    --config configs/v11_madmom.yaml \
    data.processed_root=$PROC \
    data.cpjku_fmt_root=$CPJKU_FMT \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval..."

CKPT=results/v11_madmom/best_model.pt
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls results/v11_madmom/checkpoint_epoch*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

echo "=== Eval v11-madmom on test split ==="
python -m mymodel.v11_cpjku_fullstrip.eval \
    --checkpoint $CKPT \
    --config     configs/v11_madmom.yaml \
    --split      test \
    --processed  $PROC \
    --cpjku_fmt_root $CPJKU_FMT \
    --out_dir    results/v11_madmom/eval

echo "Job finished at $(date)"
