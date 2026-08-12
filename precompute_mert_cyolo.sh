#!/bin/bash
#SBATCH --job-name=mert-cyolo
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_cyolo-%A_%a.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_cyolo-%A_%a.log

# H1 -- MERT over CYOLO's own audio, on CYOLO's own frame grid (20.0091 fps).
# 466 wavs total (353 train / 19 valid / 94 test): 6 shards, ~78 each.
#
# module load FIRST, then the venv -- +computecanada wheels link against
# module-supplied libraries and without them the dynamic loader stalls against
# CVMFS instead of raising ImportError. Guard is 7200s because a cold torch
# import took 21 minutes when the filesystem was degraded.

set -uo pipefail
echo "Job started on $(hostname) at $(date)  shard=${SLURM_ARRAY_TASK_ID}"
cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source /project/def-ichiro/pmohseni/music-alignment/.venv/bin/activate
echo "python: $(command -v python)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HOME=/scratch/pmohseni/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

time timeout 7200 python -c "import torch,transformers,librosa,soundfile;print('imports ok')" \
    || { echo "FATAL: venv imports failed or stalled >120min"; exit 1; }

DATA=/scratch/pmohseni/datasets/cyolo_data/msmd

# MERT_CYOLO_OUT / MERT_CYOLO_IR_BANK select clean vs acoustically degraded.
# The degraded bank is what lets H1 be compared against the IR-trained
# baseline: CYOLO's own ImpulseResponse transform convolves waveforms and
# cannot run once MERT is precomputed, so the degradation must be baked into
# the features (same construction as R2r_realir, which gave +11 on room).
#
#   clean:     sbatch precompute_mert_cyolo.sh
#   degraded:  MERT_CYOLO_OUT=/scratch/pmohseni/mert_emb_cyolo_ir \
#              MERT_CYOLO_IR_BANK=/scratch/pmohseni/ir_bank \
#              sbatch --export=ALL,MERT_CYOLO_OUT=...,MERT_CYOLO_IR_BANK=... precompute_mert_cyolo.sh
OUT_ROOT=${MERT_CYOLO_OUT:-/scratch/pmohseni/mert_emb_cyolo}
IR_FLAG=""
[ -n "${MERT_CYOLO_IR_BANK:-}" ] && IR_FLAG="--ir_bank ${MERT_CYOLO_IR_BANK}"
echo "out_root=$OUT_ROOT  ir=${MERT_CYOLO_IR_BANK:-none}"

for SPLIT in msmd_train msmd_valid msmd_test; do
    echo ""; echo "=== $SPLIT (shard ${SLURM_ARRAY_TASK_ID}/6) ==="
    python -m scripts.precompute_mert_cyolo \
        --wav_dir "$DATA/$SPLIT" \
        --out_dir "$OUT_ROOT/$SPLIT" \
        --shard "${SLURM_ARRAY_TASK_ID}" --num_shards 6 $IR_FLAG
done

echo ""; echo "Shard ${SLURM_ARRAY_TASK_ID} finished at $(date)"
