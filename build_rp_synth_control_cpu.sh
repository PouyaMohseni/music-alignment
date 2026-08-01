#!/bin/bash
#SBATCH --job-name=build-rp-synth
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/build_rp_synth-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/build_rp_synth-%j.log

# Matched SYNTHETIC control for the real-recording pieces -- CPJKU's own
# 'rp_synth' condition, which their Zenodo release ships alongside do/room and
# which our CB_TA-format copy of the tier lacks.
#
# Why it matters: our real-audio numbers (room 16-27%, di-left 64-71%) were
# being read against B1a's 89.2% on the 125-page synthetic msmd_test. Those are
# DIFFERENT PIECES, so the gap confounds acoustic domain shift with piece
# difficulty -- and the real-performance set is Chopin/Schumann/Mussorgsky
# heavy, i.e. exactly the repeat-heavy repertoire that already scores worst on
# synthetic audio. Rendering the SAME 25 pages with the TRAINING soundfont
# isolates the domain shift.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1; export PYTHONUNBUFFERED=1

REC=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet/data/msmd/msmd_real_performances
TRAIN_SF=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2

python -m scripts.build_acoustic_tier \
    --src "$REC" --out /scratch/pmohseni/acoustic_tiers/rp_synth \
    --soundfont "$TRAIN_SF" --tempo 1000

echo "wavs: $(ls /scratch/pmohseni/acoustic_tiers/rp_synth/performance/*.wav 2>/dev/null | wc -l)"
echo "Job finished at $(date)"
