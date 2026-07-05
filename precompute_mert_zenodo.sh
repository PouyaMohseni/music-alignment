#!/bin/bash
#SBATCH --job-name=precompute-mert-zenodo
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/precompute_mert_zenodo-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/precompute_mert_zenodo-%j.log

# B1a prerequisite: precompute frozen MERT-v1-95M embeddings for every
# (piece, tempo_factor) MIDI in the Zenodo CB_TA dataset (A0's own data),
# so training can monkey-patch midi_to_spec_otf/wav_to_spec_otf to load
# these instead of computing a live mel-spectrogram. ~6768 files total
# (945 train pages x 7 tempi + 28 val + 125 test).

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

FLUIDSYNTH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth
SOUND_FONT=third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2
OUT_ROOT=/scratch/pmohseni/mert_emb_zenodo

echo "=== train_full (945 pages x 7 tempi) ==="
python scripts/precompute_mert_zenodo.py \
    --midi_dir /scratch/pmohseni/msmd_train_full/performance \
    --out_dir  $OUT_ROOT/train_full \
    --sound_font $SOUND_FONT \
    --fluidsynth $FLUIDSYNTH

echo "=== msmd_valid ==="
python scripts/precompute_mert_zenodo.py \
    --midi_dir third_party/cpjku_unet/data/msmd/msmd_valid/performance \
    --out_dir  $OUT_ROOT/msmd_valid \
    --sound_font $SOUND_FONT \
    --fluidsynth $FLUIDSYNTH

echo "=== msmd_test ==="
python scripts/precompute_mert_zenodo.py \
    --midi_dir third_party/cpjku_unet/data/msmd/msmd_test/performance \
    --out_dir  $OUT_ROOT/msmd_test \
    --sound_font $SOUND_FONT \
    --fluidsynth $FLUIDSYNTH

echo "Job finished at $(date)"
