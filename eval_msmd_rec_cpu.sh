#!/bin/bash
#SBATCH --job-name=eval-msmd-rec
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_msmd_rec-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_msmd_rec-%j.log

# TIER 2 -- MSMD-Rec: REAL piano performances (Yamaha hybrid piano) against the
# SAME typeset MSMD scores. This is the real-audio generalisation tier
# CB_TA-Ext.md specified from the start ("zero-shot from Tier1-trained
# checkpoint") and that every result so far has been missing.
#
#   sbatch eval_msmd_rec_cpu.sh <EXPERIMENT_DIR> <room|di-left>
#   e.g. sbatch eval_msmd_rec_cpu.sh B6_impulse_response room
#
# The data was already in the repo (third_party/cpjku_unet/data/msmd/
# msmd_real_performances, 25 pages, 172MB) -- no download, no score rendering,
# no re-alignment, because MSMD-Rec reuses the standard MSMD sheet/coords.
#
# Conditions: `room` is a room microphone (true acoustic capture, the hard and
# realistic case); `di-left` is a direct pickup (clean, no room). Reporting
# both separates "does it survive a real piano" from "does it survive a real
# ROOM", which are different claims.
#
# ############################ IMPORTANT ############################
# CBEncoder checkpoints only (A0, B2..B6, C2, ...). NOT B1a / MERT_*:
# mert_patch resolves audio to a precomputed embedding keyed
# {emb_root}/{piece}_tempo_{cond}.npy, and no such file exists for real
# recordings -- a MERT run here would either crash on the missing key or, if
# a stale root were supplied, silently score SYNTHETIC audio while claiming to
# be a real-audio result. Encode the real wavs first (see
# scripts/precompute_mert_acoustic_tier.py, --tempo room).
# ###################################################################

set -uo pipefail
EXP="${1:?usage: sbatch eval_msmd_rec_cpu.sh <EXPERIMENT_DIR> <room|di-left>}"
COND="${2:-room}"

echo "Job started on $(hostname) at $(date)"
echo "experiment=$EXP  condition=$COND"
cd /project/def-ichiro/pmohseni/music-alignment

REC_DIR=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet/data/msmd/msmd_real_performances
CONFIG=/project/def-ichiro/pmohseni/music-alignment/configs/msmd_rec_${COND}.yaml
[ -f "$CONFIG" ] || { echo "FATAL: no config $CONFIG"; exit 1; }
echo "real wavs for '$COND': $(ls $REC_DIR/performance/*_${COND}.wav 2>/dev/null | wc -l)"

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

CKPT_DIR=$(find results/cb_ta_ext/$EXP/params -maxdepth 1 -name "*_$EXP" -type d 2>/dev/null | sort | tail -1)
[ -n "$CKPT_DIR" ] || { echo "FATAL: no checkpoint dir for $EXP"; exit 1; }
CKPT="$(readlink -f "$CKPT_DIR/best_model.pt")"
echo "checkpoint: $CKPT"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export LD_LIBRARY_PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/lib:${LD_LIBRARY_PATH:-}
export PATH=/scratch/pmohseni/micromamba/envs/fluidsynth/bin:${PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet
cd "$REPO/audio_conditioned_unet"

# B2/B3/B5 save their auxiliary head into the checkpoint (_ext_*), which stock
# eval_model.py rejects under strict loading ("Unexpected key(s) in
# state_dict: _ext_b5_audio_proj.*"). extensions/hooks/run_eval_native.py is
# the same wrapper eval_b5.sh uses: it applies lenient_load, then runs the
# unmodified eval_model.py. Pick it automatically rather than per-experiment,
# so this cannot be forgotten for a future aux-loss model.
ENTRY=eval_model.py
if python -c "
import sys, torch
sd = torch.load('$CKPT', map_location='cpu')
sys.exit(0 if any(k.startswith('_ext_') for k in sd) else 1)
" 2>/dev/null; then
    ENTRY=/project/def-ichiro/pmohseni/music-alignment/extensions/hooks/run_eval_native.py
    echo "checkpoint carries _ext_* keys -> using lenient_load wrapper"
fi

echo ""
echo "=== $(basename $ENTRY): $EXP on MSMD-Rec REAL audio ($COND) ==="
python "$ENTRY" \
    --param_path  "$CKPT" \
    --test_dir    "$REC_DIR" \
    --config      "$CONFIG" \
    --split_file  "$REC_DIR/rp_split.yaml" \
    --scale_factor 3 \
    --batch_size  1 \
    --seq_len     128 \
    --eval_onsets \
    --piecewise_stats

echo ""
echo "Job finished at $(date)"
