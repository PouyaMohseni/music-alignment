#!/bin/bash
#SBATCH --job-name=d1-tracker
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/d1_tracker-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/d1_tracker-%j.log

# D1 -- causal particle-filter score follower over AMT note events.
#
# NOT the project's goal: it routes through symbolic note events, which the
# end-to-end target avoids. Built as the comparison point for H1.
#
# Runs three configurations in one job, cheapest-first, because each isolates
# a different unknown:
#
#   oracle  GT MIDI as "detections" -- measures THE FILTER ALONE. If this does
#           not reach the high 90s the filter is broken and the AMT numbers
#           below mean nothing.
#   room    kong_stock transcriptions of the room microphone -- the objective.
#   di-left same take, direct pickup -- isolates what the room costs the FULL
#           pipeline, to sit alongside the ~0 F1 it costs the transcriber.
#
# Reference points on `room`: online greedy 10.73, offline (non-causal) DTW
# 98.06, our end-to-end best 56.6, cyolo_sb 79.9.
#
# venv_cpjku310 has the real madmom, which the metric path needs so note
# indices match CPJKU's bit-for-bit.  Absolute interpreter path, never
# `source activate` -- a relocated venv's activate script can repoint
# VIRTUAL_ENV at the /project venv.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"

cd /project/def-ichiro/pmohseni/music-alignment
PY=/scratch/pmohseni/venv_cpjku310/bin/python
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}

echo ""; echo "################ ORACLE (filter alone) ################"
"$PY" scripts/decomposed_tracker.py --oracle \
    --out results/analysis/d1_oracle.json

echo ""; echo "################ ROOM (kong_stock) ################"
"$PY" scripts/decomposed_tracker.py --model kong_stock --tier room \
    --out results/analysis/d1_kong_room.json

echo ""; echo "################ DI-LEFT (kong_stock) ################"
"$PY" scripts/decomposed_tracker.py --model kong_stock --tier di-left \
    --out results/analysis/d1_kong_dileft.json

echo ""
echo "Job finished at $(date)"
