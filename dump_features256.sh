#!/bin/bash
#SBATCH --job-name=feat256
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/feat256-%j.log
# The 128-dim backbone feature each candidate is scored from.
#
# Detect_17.m.0 is a 1x1 conv with 1,935 parameters, 0.13% of the model, and it
# alone produces the objectness we rank by. Its input is these 128 numbers per
# grid cell. Dumping them turns "re-fit the ranking function" into offline
# supervised learning instead of three hours of backprop per epoch.
#
# REVERBERANT training data only: ir_only is what won the validation selection
# (95.4 against ir_union's 95.1, and 91.4 against 89.8 on room), so the feature
# model is fitted on the same recipe rather than a different one.
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
unset SLURM_PROCID RANK WORLD_SIZE LOCAL_RANK
CKPT=$CY/trained_models/cyolo_sb/best_model.pt

run () { local out=$1 dir=$2 split=$3
    [ -f "$out" ] && { echo "##### $(basename $out) present"; return; }
    export DUMP_OUT="$out"; echo ""; echo "##### $(basename $out) ir=${IR_PATH:-none}"
    python extensions/hooks/run_eval_dump.py --param_path "$CKPT" \
        --test_dirs "$DATA/$dir" --split_files "$DATA/split_files/$split.yaml" \
        --only_onsets 2>&1 | stdbuf -oL grep --line-buffered -vE "it/s\]|it\]|^\s*$" | grep -E "^<= |\[DUMP\]|\[FEAT\]|\[IR\]|rror"; }

O=/scratch/pmohseni/omr/candf256; mkdir -p "$O"
unset IR_PATH
run "$O/room.npz" msmd_rp room_split                      # for the final eval
export IR_PATH=/scratch/pmohseni/ir_bank/mit_ir_survey IR_SEED=0 IR_PROB=1.0
run "$O/valid.npz" msmd_valid valid_c0_split
for i in 0 1 2 3 4 5; do run "$O/train_c$i.npz" msmd_train train_c${i}_split; done
echo ""; echo "Job finished at $(date)"; du -sh "$O"

# Train the same config on features that reach every candidate it will score.
echo ""; echo "########## refit vel_p8 on FEATK=256 features ##########"
M=/scratch/pmohseni/omr/scorer/f256; mkdir -p "$M"
export PYTHONPATH=/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
python extensions/analysis/train_cand_scorer.py --out "$M/vel_p8_f256.pt" \
    --train "$O/train_c*.npz" --valid "$O/valid.npz" \
    --use_feat --featproj 8 --seed 0 2>&1 | tail -4

echo ""; echo "########## room, both feature supplies ##########"
python - <<'PY'
import sys
import torch
torch.set_num_threads(1)
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.blend_rollout import load_with_feat, rollout
from extensions.heads.cand_scorer import load as load_ckpt
# the NEW dump gives every scored candidate a real feature, matching how the
# harness actually runs; the OLD one zero-pads past 128, which is what training
# has always seen.
for tag, dump in (('candf (top-128 features)', '/scratch/pmohseni/omr/candf/room.npz'),
                  ('candf256 (all features)', '/scratch/pmohseni/omr/candf256/room.npz')):
    try:
        pages = load_with_feat(dump)
    except Exception as e:
        print(f'{tag}: {e}'); continue
    for name, p in (('vel_p8 (trained on 128)', '/scratch/pmohseni/omr/scorer/grid/vel_p8.pt'),
                    ('vel_p8_f256 (trained on 256)', '/scratch/pmohseni/omr/scorer/f256/vel_p8_f256.pt')):
        try:
            m = load_ckpt(p)[0]
        except Exception as e:
            print(f'  {name}: {e}'); continue
        acc, n = rollout(m, pages, blend=0.7)
        print(f'  {tag:26s} {name:30s} room {acc:6.2f}', flush=True)
PY
