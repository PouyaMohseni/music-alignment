#!/bin/bash
#SBATCH --job-name=mert-aug
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=11:00:00
#SBATCH --array=0-39
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_aug-%A_%a.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_aug-%A_%a.log

# R2 -- build the ACOUSTICALLY DEGRADED MERT embedding bank for the 6615
# training performances (see scripts/precompute_mert_augmented.py for why this
# is done ahead of time rather than as an on-the-fly transform: MERT is frozen
# and consumed from .npy, so waveform augmentation is otherwise invisible).
#
# CPU array, not GPU: MERT-v1-95M is small, this is embarrassingly parallel
# over pieces, and the GPU queue is the scarce resource that the four training
# tracks need. 40 shards x ~165 pieces each.
#
# Output goes to /scratch (9.4 GB, matching the clean bank) -- NOT /project,
# which is at ~96% of its 500k inode quota.
#
# Idempotent: every shard skips keys whose .npy already exists, and the
# degradation seed is derived from the piece key, so a requeued or re-run
# shard reproduces the same audio instead of creating a second condition.

set -uo pipefail
echo "Job started on $(hostname) at $(date)  shard=${SLURM_ARRAY_TASK_ID}"
cd /project/def-ichiro/pmohseni/music-alignment

# Do NOT `source .../activate`. /scratch/pmohseni/music-alignment-venv/bin/activate
# HARDCODES VIRTUAL_ENV=/lustre06/project/.../.venv (pyvenv.cfg shows it was
# created from there), so sourcing it silently puts the /project venv back on
# PATH -- exactly the thing being avoided. Job 51150 proved this: the script
# said /scratch and the canary still printed a /project interpreter.
#
# Why /project must be avoided: torch+transformers is ~50k small files and
# /project is at ~96% of its 500k INODE quota, so metadata-heavy reads stall.
# Pilot 50660 sat 10+ min in Lustre cl_sync_io_wait at 0 CPU seconds, never
# executing. /scratch is the same Lustre but not inode-starved.
#
# Setting VIRTUAL_ENV/PATH by hand is all activate does; python then resolves
# its own prefix from the executable location + pyvenv.cfg (verified: prefix
# and site-packages both land under /scratch).
VENV=/scratch/pmohseni/music-alignment-venv
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME

echo "python: $(command -v python)"
timeout 900 python -c "import sys; print('prefix:', sys.prefix)" || { echo "FATAL: venv unusable/stalled"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/pmohseni/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

timeout 900 python -c "import torch, transformers, librosa, soundfile; print('imports ok')" \
    || { echo "FATAL: venv imports failed or stalled >15min"; exit 1; }

MIDI_DIR=/scratch/pmohseni/msmd_train_full/performance
OUT_DIR=/scratch/pmohseni/mert_emb_aug/train_full
SF=third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2
FS=/scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth

python -m scripts.precompute_mert_augmented \
    --midi_dir "$MIDI_DIR" \
    --out_dir  "$OUT_DIR" \
    --sound_font "$SF" \
    --fluidsynth "$FS" \
    --shard "${SLURM_ARRAY_TASK_ID}" --num_shards 40 \
    --tilt --ir --noise

echo "Shard ${SLURM_ARRAY_TASK_ID} finished at $(date)"
