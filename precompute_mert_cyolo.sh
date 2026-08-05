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
for SPLIT in msmd_train msmd_valid msmd_test; do
    echo ""; echo "=== $SPLIT (shard ${SLURM_ARRAY_TASK_ID}/6) ==="
    python -m scripts.precompute_mert_cyolo \
        --wav_dir "$DATA/$SPLIT" \
        --out_dir /scratch/pmohseni/mert_emb_cyolo/$SPLIT \
        --shard "${SLURM_ARRAY_TASK_ID}" --num_shards 6
done

echo ""; echo "Shard ${SLURM_ARRAY_TASK_ID} finished at $(date)"
