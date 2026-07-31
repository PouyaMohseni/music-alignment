#!/bin/bash
#SBATCH --job-name=smoke-temporal-arch
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/smoke_temporal_arch-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/smoke_temporal_arch-%j.log

# Correctness gate for the N1/N2/N3 temporal-architecture experiments before
# any of them is given GPU time. Chiefly asserts that N2 and N3 are EXACTLY
# equal to stock CB_TA at initialisation, which is what makes warm-starting
# them from B1a's converged checkpoint meaningful.
#
# Submitted as a batch job rather than run on the login node: a first attempt
# on narval1 wedged at 0% CPU, and per memory/cluster_workflow.md (2026-07-22
# Acceptable-Use incident) anything beyond a trivial shell command belongs in
# a job. CPU-only -- this is a shape/identity test, no GPU needed.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment

SETUP_LOCK=/project/def-ichiro/pmohseni/music-alignment/.cpjku_submodule_setup.flock
(
    flock -w 120 200
    if [ ! -f third_party/cpjku_unet/audio_conditioned_unet/network.py ]; then
        git submodule update --init third_party/cpjku_unet || true
    fi
    git -C third_party/cpjku_unet checkout ismir-2020
) 200>"$SETUP_LOCK"

module load gcc opencv
source /scratch/pmohseni/venv_cpjku310/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u -m scripts.smoke_test_temporal_arch
STATUS=$?

echo ""
echo "smoke test exit status: $STATUS"
echo "Job finished at $(date)"
exit $STATUS
