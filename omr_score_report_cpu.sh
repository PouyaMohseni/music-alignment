#!/bin/bash
#SBATCH --job-name=omr-score
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/pmohseni/omr/logs/omr-score-%j.log
#SBATCH --error=/scratch/pmohseni/omr/logs/omr-score-%j.log
#
# Score the oemer detections from jobs 433842 (native) and 433844 (hires)
# against the MSMD MUNG ground truth, and render the report + failure crops.
#
# Detection already ran: 20/20 pages per variant, rc=0, ~5 min/page.  This step
# is pure numpy matching over the per-page JSON, so it is cheap -- but it still
# goes through sbatch rather than the login node.
#
# Same environment constraints as the detection job: venv on /scratch (the
# /project inode quota is ~96% full), absolute interpreter path (never `source
# activate`, which can repoint VIRTUAL_ENV at the /project venv), and cv2 from
# the system module because the pip wheel is a stub here.  The report step
# renders annotated crops, so it needs cv2 too.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"

WORK=/scratch/pmohseni/omr
PY=/scratch/pmohseni/venv_oemer/bin/python

module load python/3.11
module load gcc/12.3 opencv/4.11.0

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/analysis "$WORK/crops"

for VARIANT in native hires; do
    echo
    echo "################ $VARIANT ################"
    SCORED="$WORK/out/${VARIANT}_scored.json"

    "$PY" scripts/omr_score.py \
        --work "$WORK" --variant "$VARIANT" --out "$SCORED" || continue

    "$PY" scripts/omr_report.py \
        --scored "$SCORED" --work "$WORK" --variant "$VARIANT" \
        --crops "$WORK/crops/$VARIANT" || true

    # keep the small summary in the repo; bulk detections stay on /scratch
    cp -f "$SCORED" "results/analysis/omr_${VARIANT}_scored.json" 2>/dev/null || true
done

echo
echo "Job finished at $(date)"
