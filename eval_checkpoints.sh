#!/bin/bash
#SBATCH --job-name=ckpts
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/ckpts-%j.log
# Run the decode on the checkpoints we have NOT been using.
#
# The repo ships three: cyolo (71.2 room), cyolo_sb (79.9) and cyolo_sb_a (86.5).
# Every result in this project so far improves cyolo_sb. cyolo_sb_a was written
# off as "not reproducible" -- true of RETRAINING it, since the +A augmentation
# data is absent from the Zenodo release, but its weights are published and our
# method trains nothing. We have been improving the weaker model by choice we
# never actually made.
#
# If the decode is a property of the readout rather than of one checkpoint, it
# should transfer. If it only works on cyolo_sb, that is worth knowing too --
# it would mean we fitted the decode to one model's failure modes.
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
REC=/scratch/pmohseni/omr/ckpts; mkdir -p "$REC"
export SEARCH_KIND=beam BEAM=1 C2_FWD=6.0 C2_SIGMA=18.0 C2_LAM=1.0 C2_JUMP=-6.0
export CLUSTER_PX=0 C2_TOPK=100000 TIME_SIG_POW=0 TIME_REF=5
export ANCHOR=start WINDOW=0 Z_MASK=none ORACLE=0

run () { local ck=$1 tier=$2 arm=$3
    export REC_OUT="$REC/${ck}_${tier}_${arm}.npz"
    echo ""; echo "##### ckpt=$ck  tier=$tier  arm=$arm"
    python extensions/hooks/run_eval_search.py \
        --param_path "$CY/trained_models/$ck/best_model.pt" \
        --test_dirs "$DATA/msmd_rp" --split_files "$DATA/split_files/${tier}_split.yaml" \
        --only_onsets 2>&1 | grep -vE "it/s\]|it\]|^\s*$" \
        | grep -E "^<= |^Average accuracy|rror|Traceback"; }

for CK in cyolo_sb_a cyolo; do
  for TIER in room do rp_synth; do
    export C2_CLASSES='' TIME_MU_POW=0;  run "$CK" "$TIER" baseline
    export C2_CLASSES='0' TIME_MU_POW=0; run "$CK" "$TIER" decode
    export C2_CLASSES='0' TIME_MU_POW=1; run "$CK" "$TIER" timeaware
  done
done
echo ""; echo "Job finished at $(date)"
