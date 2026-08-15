#!/bin/bash
#SBATCH --job-name=probe-suite
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/probe_suite-%j.log
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
REC=/scratch/pmohseni/omr/probe; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0

run () {  # tag  [extra env already exported]
    export REC_OUT="$REC/$1_room.npz"
    echo ""; echo "##### $1  drop=[${DROP_SCALES:-}] film=${FILM_SCALE:-1.0} sys=${SYS_SLACK:-0} sw=${SW:-416}"
    python extensions/hooks/run_eval_search.py --param_path "$CKPT" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/room_split.yaml" \
        --scale_width ${SW:-416} --only_onsets 2>&1 \
        | grep -vE "it/s\]|it\]" | grep -E "^<= 0.5|^Average accuracy for (Bar|System)|[Ee]rror|Traceback|^\\s+File |ModuleNotFound|RuntimeError|\\[PROBE\\]"
}
reset_env () { export DROP_SCALES="" FILM_SCALE=1.0 SYS_SLACK=0 SW=416; }

reset_env; run control                       # must reproduce 84.7

# --- BIG: does it actually use the audio?
reset_env; export FILM_SCALE=0.0;  run film_off
reset_env; export FILM_SCALE=0.5;  run film_half
reset_env; export FILM_SCALE=1.5;  run film_1p5
reset_env; export FILM_SCALE=2.0;  run film_2x

# --- BIG: which detection scale finds the note?
reset_env; export DROP_SCALES=0;   run drop_P3
reset_env; export DROP_SCALES=1;   run drop_P4
reset_env; export DROP_SCALES=2;   run drop_P5
reset_env; export DROP_SCALES=1,2; run only_P3
reset_env; export DROP_SCALES=0,2; run only_P4
reset_env; export DROP_SCALES=0,1; run only_P5

# --- BIG: input resolution
reset_env; export SW=320;  run sw320
reset_env; export SW=512;  run sw512
reset_env; export SW=640;  run sw640

# --- MEDIUM: pin the note to the predicted system (acc 0.917)
reset_env; export SYS_SLACK=5;   run sys5
reset_env; export SYS_SLACK=20;  run sys20
reset_env; export SYS_SLACK=50;  run sys50
echo ""; echo "Job finished at $(date)"
