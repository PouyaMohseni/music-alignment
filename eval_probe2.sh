#!/bin/bash
#SBATCH --job-name=probe2
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/probe2-%j.log
# Take the frozen model apart: which parts carry the signal?
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
CY=/scratch/pmohseni/datasets/cyolo_score_following
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
export CYOLO_ROOT=$CY
export PYTHONPATH=$CY:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
CKPT=$CY/trained_models/cyolo_sb/best_model.pt
REC=/scratch/pmohseni/omr/probe2; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0

run () {  # tag  [extra env already exported]
    export REC_OUT="$REC/$1_room.npz"
    echo ""; echo "##### $1  drop=[${DROP_SCALES:-}] film=${FILM_SCALE:-1.0} sys=${SYS_SLACK:-0} bar=${BAR_SLACK:-0} z=${Z_MASK:-none} sw=${SW:-416}"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --scale_width ${SW:-416} --only_onsets 2>&1 \
        | grep -vE "it/s\]|it\]" | grep -E "^<= 0.5|^Average accuracy for (Bar|System)|[Ee]rror|Traceback|^\\s+File |ModuleNotFound|RuntimeError|\\[PROBE\\]"
}
reset_env () { export DROP_SCALES="" FILM_SCALE=1.0 SYS_SLACK=0 BAR_SLACK=0 Z_MASK=none SW=416; }

reset_env; run control                        # must reproduce 84.7

# --- does the recurrent path carry anything? z = z_enc(cat(history_64, now_32))
reset_env; export Z_MASK=hist; run z_no_history   # keep the last 2 s only
reset_env; export Z_MASK=now;  run z_no_present   # keep the LSTM state only

# --- pin the note to the predicted BAR (finer region than the system)
reset_env; export BAR_SLACK=5;   run bar5
reset_env; export BAR_SLACK=20;  run bar20
reset_env; export BAR_SLACK=50;  run bar50

# --- bar AND system together
reset_env; export BAR_SLACK=20; export SYS_SLACK=20; run bar20_sys20
echo ""; echo "Job finished at $(date)"
