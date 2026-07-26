#!/bin/bash
#SBATCH --job-name=v11-mert-finetune
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --exclude=ng[11105-11106,31001]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v11_mert_finetune-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v11_mert_finetune-%j.log

# Same CB_TA-faithful architecture as v11/v11-madmom, but the audio encoder
# is a LIVE, fine-tunable MERT-v1-95M (mymodel/v10_mert_unet/mert_live.py)
# instead of a static spectrogram. Every other MERT usage in this project
# (v13/v14/v15, B1a) reads precomputed, FROZEN embeddings -- MERT's
# self-supervised pretraining was never optimized for this task's
# 50ms-precision localization, and a frozen readout can't adapt that. This
# is the first experiment where MERT's own weights get gradient from the
# alignment task (smoke-tested first: scripts/smoke_test_mert_finetune.py,
# job 65737515, confirmed all 85M unfrozen encoder params receive nonzero
# gradient on a real forward+backward pass before committing to this run).
#
# Expect this to be much slower per epoch than v11-madmom (~2.3h/epoch):
# one 64-frame BPTT chunk took ~4.6s forward+backward in the smoke test
# (MERT is a 95M-param transformer, not a small CNN), so a full epoch over
# 354 pieces could plausibly take many hours to low-single-digit days.
# Auto-resumes across multiple 24h submissions like every other long-running
# job in this project.

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v11_mert_finetune

module load gcc opencv
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# job 65737815 OOMed 11 min in ("4.28 GiB reserved by PyTorch but
# unallocated" -- a fragmentation signature, not a fixed too-large
# allocation): BPTT with seq_len=64 across pieces of widely varying
# score-image width creates a very uneven allocation pattern while
# fine-tuning all 85M MERT params. expandable_segments is PyTorch's own
# suggested fix for exactly this error; seq_len halved as an added margin.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# job 65852778 (seq_len=32 + expandable_segments) OOMed again anyway --
# "38.42 GiB memory in use" of a 39.49 GiB A100, a genuine too-large
# allocation this time (not fragmentation): fully fine-tuning all 85M MERT
# encoder params retains activations through the whole stack for BPTT
# backward. Applied the config's own documented fallback: freeze all but
# the top 4 MERT layers.
#
# job 66338467 (seq_len=32 + unfreeze_last_n=4) OOMed AGAIN anyway --
# 39.18/39.49 GiB, barely different from before. The traceback this time
# points INTO the U-Net DECODER's conv2d (mymodel/v9_cpjku/cpjku_network.py
# forward), not into MERT -- freezing 8/12 MERT layers was a real but
# insufficient fix, because the DOMINANT memory cost is actually the U-Net's
# own activations retained across all 32 BPTT timesteps simultaneously for
# backward, not MERT's transformer stack. Halving seq_len again (32->16)
# directly cuts that dominant cost, unlike freezing MERT layers further.

PROC=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/processed

echo "=== Train v11-mert-finetune: CB_TA architecture + live fine-tuned MERT-v1-95M ==="
RESUME_FLAG=""
if ls results/v11_mert_finetune/checkpoint_epoch*.pt 1>/dev/null 2>&1; then
    echo "Found existing checkpoint — resuming."
    RESUME_FLAG="--resume"
fi
python -m mymodel.v11_mert_finetune.train \
    --config configs/v11_mert_finetune.yaml \
    data.processed_root=$PROC \
    train.seq_len=16 \
    mert.unfreeze_last_n=4 \
    $RESUME_FLAG

echo "Training finished at $(date). Running eval..."

CKPT=results/v11_mert_finetune/best_model.pt
if [ ! -f "$CKPT" ]; then
    CKPT=$(ls results/v11_mert_finetune/checkpoint_epoch*.pt 2>/dev/null | sort | tail -1)
fi
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint found after training"; exit 1
fi
echo "Evaluating: $CKPT"

echo "=== Eval v11-mert-finetune on test split ==="
python -m mymodel.v11_mert_finetune.eval \
    --checkpoint $CKPT \
    --config     configs/v11_mert_finetune.yaml \
    --split      test \
    --processed  $PROC \
    --out_dir    results/v11_mert_finetune/eval

echo "Job finished at $(date)"
