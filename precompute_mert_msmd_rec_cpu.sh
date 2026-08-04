#!/bin/bash
#SBATCH --job-name=mert-msmd-rec
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_msmd_rec-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_msmd_rec-%j.log

# Encode MERT embeddings from MSMD-Rec's REAL recordings, so B1a / MERT_*
# checkpoints can be evaluated on Tier 2 at all.
#
# This is mandatory, not an optimisation. extensions/hooks/mert_patch.py never
# encodes audio at eval time -- `_load_mert_spec` resolves a piece to
# {emb_root}/{piece}_tempo_{cond}.npy. No such file exists for the real
# recordings, so a MERT model pointed at MSMD-Rec would either die on the
# missing key or, if handed a stale embeddings root, quietly score the
# SYNTHETIC audio while the log claims a real-audio result. That second
# failure is silent and would put a false number in the paper.
#
# CPU-only: MERT-v1-95M is small, the tier is 25 pages (~20 min of audio), and
# the GPU queue is saturated while CPU is not. Runs both recording conditions.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
# .venv (py3.11), NOT venv_cpjku310 (py3.10). venv_cpjku310 exists for the
# CPJKU eval path because it has REAL madmom, but it has no librosa and no
# transformers, so the first attempt (job 66882934) died in 2 minutes on
# `import librosa`. .venv has librosa 0.11 + transformers 5.9 + torch 2.11 and
# is what precompute_mert_zenodo.py needs -- which also means it is what
# produced the TRAINING embeddings, so encoding the real recordings here keeps
# them consistent with the features the checkpoints were trained on. Nothing
# in this script touches madmom.
# module load FIRST, then the venv. This is not optional and its absence does
# not fail loudly: Compute Canada's `+computecanada` wheels link against
# libraries supplied by loaded modules, and without them the dynamic loader
# stalls resolving against CVMFS -- the process sits in cl_sync_io_wait at
# 00:00:00 CPU forever rather than raising ImportError.
#
# Pilots 50660 and 52401 both hung exactly that way. I first blamed /project's
# inode quota and moved the venv to /scratch; 52401 proves that was wrong --
# it hung identically on /scratch. The real difference is that
# precompute_mert_zenodo.sh, which successfully built the CLEAN 6615-piece
# bank, does `module load gcc opencv` and my script did not.
#
# .venv is also deliberate, not incidental: the clean bank was encoded with it,
# and the augmented bank must come from the SAME MERT stack or a clean-vs-
# augmented comparison is confounded by library versions.
module load gcc opencv
source /project/def-ichiro/pmohseni/music-alignment/.venv/bin/activate

echo "python: $(command -v python)"
timeout 900 python -c "import sys; print('prefix:', sys.prefix)" || { echo "FATAL: venv unusable/stalled"; exit 1; }
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/pmohseni/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REC_DIR=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet/data/msmd/msmd_real_performances

for COND in room di-left; do
    echo ""
    echo "=== encoding MSMD-Rec condition: $COND ==="
    python -m scripts.precompute_mert_acoustic_tier \
        --tier_dir "$REC_DIR" \
        --out_dir  /scratch/pmohseni/mert_emb_msmd_rec/$COND \
        --tempo    "$COND"
done

echo ""
echo "=== summary ==="
for COND in room di-left; do
    echo "$COND: $(ls /scratch/pmohseni/mert_emb_msmd_rec/$COND/*.npy 2>/dev/null | wc -l) embeddings"
done
echo "Job finished at $(date)"
