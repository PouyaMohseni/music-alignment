#!/bin/bash
#SBATCH --job-name=watch-ep20
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/watch_ep20-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/watch_ep20-%j.log

# CPU-only watcher: polls for epoch-20 checkpoints and fires GPU eval jobs.

echo "Watcher started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment

declare -A CKPTS=(
    [v13]="/scratch/pmohseni/results/v13_mert_linear/checkpoint_epoch020.pt"
    [v14]="/scratch/pmohseni/results/v14_mert_bilstm/checkpoint_epoch020.pt"
    [v15]="/scratch/pmohseni/results/v15_mert_mlp/checkpoint_epoch020.pt"
)
declare -A EVAL_SCRIPTS=(
    [v13]="eval_v13_ep20.sh"
    [v14]="eval_v14_ep20.sh"
    [v15]="eval_v15_ep20.sh"
)

submitted_v13=0
submitted_v14=0
submitted_v15=0

while true; do
    for v in v13 v14 v15; do
        varname="submitted_${v}"
        if [ "${!varname}" -eq 0 ] && [ -f "${CKPTS[$v]}" ]; then
            echo "[$(date +%H:%M:%S)] Found ${CKPTS[$v]} — submitting ${EVAL_SCRIPTS[$v]}"
            sbatch "${EVAL_SCRIPTS[$v]}"
            eval "submitted_${v}=1"
        fi
    done

    [ "$submitted_v13" -eq 1 ] && [ "$submitted_v14" -eq 1 ] && [ "$submitted_v15" -eq 1 ] && break
    sleep 120
done

echo "All epoch-20 eval jobs submitted at $(date)"
