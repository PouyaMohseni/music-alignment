#!/bin/bash
#SBATCH --job-name=gen-perf-midis
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/gen_perf_midis-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/gen_perf_midis-%j.log

# Generate ALL 945x7=6615 tempo-augmented performance MIDIs to /scratch to avoid
# the /project file-count quota (500k files).  Builds a self-contained dataset dir
# on scratch with symlinked score/ files and generated performance/ MIDIs.
# After this completes, train_cpjku_paper_CB_TA.sh uses --train_set pointing to scratch.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"

source /scratch/pmohseni/venv_cpjku310/bin/activate

PROJECT_TRAIN=/project/def-ichiro/pmohseni/music-alignment/third_party/cpjku_unet/data/msmd/msmd_train
SCRATCH_TRAIN=/scratch/pmohseni/msmd_train_full

# Build scratch dir: symlink score/ from project, generate performance/ to scratch
mkdir -p "$SCRATCH_TRAIN/performance"
if [ ! -L "$SCRATCH_TRAIN/score" ]; then
    ln -s "$PROJECT_TRAIN/score" "$SCRATCH_TRAIN/score"
fi

cd /project/def-ichiro/pmohseni/music-alignment

echo "Generating all performance MIDIs to $SCRATCH_TRAIN/performance/ ..."
python generate_msmd_train_perf_midis.py \
    --data_dir "$PROJECT_TRAIN" \
    --out_perf_dir "$SCRATCH_TRAIN/performance"

echo ""
echo "Files in scratch performance/: $(ls $SCRATCH_TRAIN/performance/*.mid 2>/dev/null | wc -l)"
echo "Done at $(date)"
