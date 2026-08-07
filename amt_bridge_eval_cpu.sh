#!/bin/bash
#SBATCH --job-name=amt-bridge-eval
#SBATCH --account=def-ichiro

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/amt_bridge_eval-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/amt_bridge_eval-%j.log

# Score the AMT bridge on the transcriptions job 402243 produced
# (/scratch/pmohseni/amt_out/{kong_stock,edwards_robust}/{room,di-left}, 25 pages each).
#
# Two numbers come out of this:
#   1. mir_eval note-onset F1 per (model x tier).  room and di-left are two
#      microphones on the SAME take, so room-minus-di-left isolates exactly what
#      the room costs with the performance held fixed.
#   2. end-to-end pct@0.5s through the CPJKU harness metric, so it sits on the
#      same axis as our image models (45.6 room) and cyolo_sb (79.9).
#
# Uses venv_cpjku310 by ABSOLUTE INTERPRETER PATH, not `source activate`:
# /scratch/pmohseni/music-alignment-venv/bin/activate hardcodes VIRTUAL_ENV back
# to the /project venv, so activating a relocated venv silently gives you the
# wrong interpreter.  This venv is the one with REAL madmom, which the script
# needs so its MIDI note indices match CPJKU's bit-for-bit.
#
# --shift_diag grid-searches a constant wav-vs-MIDI offset per piece.  Worth the
# extra minutes: an IR convolution bug already cost us 4-20 frames of label
# desync once, and a nonzero median shift here would mean the transcriptions are
# being scored against misaligned ground truth.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment

PY=/scratch/pmohseni/venv_cpjku310/bin/python
export OMP_NUM_THREADS=4
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}

"$PY" scripts/amt_bridge_eval.py \
    --amt_root /scratch/pmohseni/amt_out \
    --out results/amt_bridge_eval.json \
    --shift_diag

echo "Job finished at $(date)"
