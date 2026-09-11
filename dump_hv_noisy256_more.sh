#!/bin/bash
#SBATCH --job-name=hv256m
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/hv256m%a-%A.log
# The two harder noise tiers at FEATK=256, one per array task.
#
# The config and the checkpoint were both chosen on candhv256/valid_snr12, the
# supply the harness gives the scorer at inference. The noise-degradation table
# on the demo page was built from the FEATK=128 dumps, which is fine for vel_p8
# (trained and scored at 128) but would put a different nbrp64_s0 number in its
# 12 dB row than the one the selection used. Only 12 and 6 dB exist at 256, so
# 3 and 0.5 dB are made here with the same recipe as dump_hv_noisy256.sh.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 DUMP_MAXK=256 DUMP_FEATK=256
export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_PROB=1.0 IR_SEED=7
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
O=/scratch/pmohseni/omr/candhv256; mkdir -p "$O"
SNRS=(3 0.5); SNR=${SNRS[$SLURM_ARRAY_TASK_ID]}
T="$O/valid_snr$SNR.npz"
if [ -f "$T" ] && [ "$(stat -c %s "$T")" -gt 1000000 ]; then
    echo "##### snr$SNR present ($(stat -c %s "$T") bytes)"; exit 0
fi
export IR_SNR=$SNR DUMP_OUT="$T"
echo "##### held-out 80 pieces, room + noise at ${SNR} dB"
python extensions/hooks/run_eval_dump.py \
    --param_path "$CY/trained_models/cyolo_sb/best_model.pt" \
    --test_dirs "$DATA/msmd_train" --split_files "$DATA/split_files/hv_valid_split.yaml" \
    --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" \
    | grep -E "^<= |\[DUMP\]|\[IR\]|rror|Traceback"
ls -l "$T"
echo "Job finished at $(date)"
