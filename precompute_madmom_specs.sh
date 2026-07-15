#!/bin/bash
#SBATCH --job-name=precompute-madmom-specs
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/precompute_madmom_specs-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/precompute_madmom_specs-%j.log

# Root-cause fix for the eval-harness discrepancy found this session: the
# frozen, unmodified official CB_TA checkpoint scores ~85% via the paper's
# own eval_model.py (real madmom, venv_cpjku310) but only 15.1% via our
# mymodel.cpjku_adapter.eval_official (librosa filterbank approximation,
# main .venv, since madmom doesn't install on Python>=3.11). That same
# eval_official.py path is what evaluates b1a-b6/c2/cpjku_aug's own trained
# checkpoints too, so their historically low scores are likely contaminated
# by this same mismatch, not (only) reflecting the architectural ideas being
# tested.
#
# This script runs under venv_cpjku310 (real madmom, Python 3.10) and caches
# the exact training-side spectrogram (wav_to_spec_otf) to
# data/MSMD/cpjku_fmt/spec_madmom/<piece_id>.npy. eval_official.py now checks
# this cache first and only falls back to the librosa approximation (with a
# loud warning) if missing.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv python/3.10
source /scratch/pmohseni/venv_cpjku310/bin/activate

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

CPJKU_FMT=/project/def-ichiro/pmohseni/music-alignment/data/MSMD/cpjku_fmt

echo "=== Precomputing real-madmom spectrograms (test split performances) ==="
python -m mymodel.cpjku_adapter.precompute_madmom_specs \
    --cpjku_root third_party/cpjku_unet \
    --cpjku_data $CPJKU_FMT

echo "Job finished at $(date)"
