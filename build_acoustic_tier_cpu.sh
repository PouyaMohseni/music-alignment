#!/bin/bash
#SBATCH --job-name=build-acoustic-tier
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/build_acoustic_tier-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/build_acoustic_tier-%j.log

# Builds three acoustic test tiers from the MSMD test set. Same sheet images,
# same notehead coords, same note timing -- only the audio changes -- so this
# isolates acoustic-domain shift without needing any new score alignment
# (which is exactly what blocks ASAP; see scripts/build_acoustic_tier.py).
#
#   ctrl    training soundfont, no degradation
#   timbre  DIFFERENT piano soundfont
#   room    different piano + convolutional reverb + pink noise @ 20 dB SNR
#
# `ctrl` is not padding -- it is the control that validates the harness. It
# routes the SAME audio content through CPJKU's real_perf=True path (read a
# .wav) instead of the usual synthesise-the-MIDI path. If ctrl does not
# reproduce the synthetic score (~89.2% for B1a), then the real_perf plumbing
# is wrong and every timbre/room number would be measuring our own bug rather
# than acoustic robustness. Check ctrl FIRST.
#
# Output goes to /scratch: /project is at ~96% of its 500k inode quota and
# these tiers add ~375 wav files. score/ is symlinked, never copied.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

TRAIN_SF=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2
ALT_SF=/scratch/pmohseni/venv_cpjku310/lib/python3.10/site-packages/pretty_midi/TimGM6mb.sf2
if [ ! -f "$ALT_SF" ]; then
    ALT_SF=$(find /scratch/pmohseni -name 'TimGM6mb.sf2' 2>/dev/null | head -1)
fi
echo "train soundfont: $TRAIN_SF"
echo "alt   soundfont: $ALT_SF"
[ -f "$ALT_SF" ] || { echo "FATAL: no alternate soundfont found"; exit 1; }

BASE=/scratch/pmohseni/acoustic_tiers

echo ""
echo "=== tier 1/3: ctrl (training soundfont, no degradation) -- HARNESS CONTROL ==="
python -m scripts.build_acoustic_tier --out "$BASE/ctrl" --soundfont "$TRAIN_SF"

echo ""
echo "=== tier 2/3: timbre (different piano) ==="
python -m scripts.build_acoustic_tier --out "$BASE/timbre" --soundfont "$ALT_SF"

echo ""
echo "=== tier 3/3: room (different piano + reverb + pink noise @20dB) ==="
python -m scripts.build_acoustic_tier --out "$BASE/room" --soundfont "$ALT_SF" --ir --snr-db 20

echo ""
echo "=== summary ==="
for t in ctrl timbre room; do
    echo "$t: $(ls $BASE/$t/performance/*.wav 2>/dev/null | wc -l) wavs, $(du -sh $BASE/$t 2>/dev/null | cut -f1)"
done
echo "Job finished at $(date)"
