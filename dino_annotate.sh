#!/bin/bash
#SBATCH --job-name=dinoann
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/dinoann_%a-%A.log
# DINOv2 features into the candidate dumps (after dino_maps.sh).
#
#   sbatch dino_annotate.sh pca                      fit the 768 -> 128 PCA once
#   sbatch --array=0-7 dino_annotate.sh cyolo        cyolo_sb dumps (candf, candf256 room)
#   sbatch --array=0-7 dino_annotate.sh h1           MERT-detector dumps (candh1)
#
# The PCA is fitted on cyolo_sb's training candidates only and applied to every
# dump, both detectors, so the projection never sees a test page.
set -uo pipefail
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc python/3.10 opencv/4.10.0
source /scratch/pmohseni/venv_cyolo/bin/activate
export PYTHONPATH=/scratch/pmohseni/datasets/cyolo_score_following:/project/def-ichiro/pmohseni/music-alignment:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
O=/scratch/pmohseni/omr
P=$O/dino_pca.npz
case "$1" in
  pca)
    python extensions/analysis/dino_annotate.py fit --out "$P" --src "$O"/candf/train_c*.npz
    exit $? ;;
  cyolo) SRC=("$O/candf256/room.npz" "$O/candf/valid.npz"); B=candf;  DST=$O/candf_dino ;;
  h1)    SRC=("$O/candh1/room.npz" "$O/candh1/valid.npz");  B=candh1; DST=$O/candh1_dino ;;
  *) echo "usage: dino_annotate.sh pca|cyolo|h1"; exit 1 ;;
esac
for c in 0 1 2 3 4 5; do SRC+=("$O/$B/train_c$c.npz"); done
S=${SRC[$SLURM_ARRAY_TASK_ID]}
[ -f "$DST/$(basename "$S")" ] && { echo "present: $DST/$(basename "$S")"; exit 0; }
python extensions/analysis/dino_annotate.py annotate --pca "$P" --src "$S" --dst "$DST/$(basename "$S")"
