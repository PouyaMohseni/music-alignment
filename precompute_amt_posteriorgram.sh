#!/bin/bash
#SBATCH --job-name=amt-post
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/amt_post-%A_%a.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/amt_post-%A_%a.log

# A1 -- AMT posteriorgram bank over CYOLO's audio, on CYOLO's frame grid.
#
# Motivated directly by our own measurement: the room costs this transcriber
# 0.001 onset F1 (0.9116 room vs 0.9124 direct pickup, same take) while it costs
# our trackers ~30 points. We take the room-invariant REPRESENTATION and leave
# the note events behind -- what is stored is the raw continuous 88-band
# posterior, nothing thresholded or decoded, so no symbolic intermediate enters
# the model.
#
#   clean:     sbatch precompute_amt_posteriorgram.sh
#   degraded:  AMT_POST_OUT=... AMT_POST_WAV_SUFFIX=... (see A1 notes)
#
# 466 wavs (353 train / 19 valid / 94 test) over 6 shards. CPU is fine: the
# model is small and this is embarrassingly parallel, and the GPU queue is the
# scarce resource the training tracks need.

set -uo pipefail
echo "Job started on $(hostname) at $(date)  shard=${SLURM_ARRAY_TASK_ID}"
cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source /project/def-ichiro/pmohseni/music-alignment/.venv/bin/activate
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}

time timeout 7200 python -c "import torch,librosa,soundfile,piano_transcription_inference;print('imports ok')" \
    || { echo "FATAL: venv imports failed or stalled"; exit 1; }

CKPT=${AMT_CKPT:-/scratch/pmohseni/amt_ckpts/kong_stock.pth}
[ -s "$CKPT" ] || { echo "FATAL: checkpoint missing: $CKPT"; exit 1; }
OUT_ROOT=${AMT_POST_OUT:-/scratch/pmohseni/amt_post_cyolo}
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd

for SPLIT in msmd_train msmd_valid msmd_test; do
    echo ""; echo "=== $SPLIT (shard ${SLURM_ARRAY_TASK_ID}/6) ==="
    python -m scripts.precompute_amt_posteriorgram \
        --wav_dir "$DATA/$SPLIT" \
        --out_dir "$OUT_ROOT/$SPLIT" \
        --checkpoint "$CKPT" --device cpu \
        --shard "${SLURM_ARRAY_TASK_ID}" --num_shards 6
done

echo ""; echo "Shard ${SLURM_ARRAY_TASK_ID} finished at $(date)"
