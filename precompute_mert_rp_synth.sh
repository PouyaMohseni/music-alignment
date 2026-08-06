#!/bin/bash
#SBATCH --job-name=mert-rpsynth
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mert_rp_synth-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/mert_rp_synth-%j.log

# MERT bank for the rp_synth control tier (25 wavs, {piece}_1000.wav).
# Needed before ANY MERT-family model can be evaluated on the control -- without
# it mert_patch cannot resolve a piece to an embedding.
#
# module load FIRST then .venv: +computecanada wheels link against
# module-supplied libs and the loader otherwise stalls against CVMFS instead of
# raising ImportError. Guard is generous because a cold torch import took 21 min
# when the filesystem was degraded.

set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /project/def-ichiro/pmohseni/music-alignment/.venv/bin/activate
echo "python: $(command -v python)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HOME=/scratch/pmohseni/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

time timeout 7200 python -c "import torch,transformers,librosa,soundfile;print('imports ok')" \
    || { echo "FATAL: venv imports failed or stalled"; exit 1; }

python -m scripts.precompute_mert_acoustic_tier \
    --tier_dir /scratch/pmohseni/acoustic_tiers/rp_synth \
    --out_dir  /scratch/pmohseni/mert_emb_rp_synth \
    --tempo    1000

N=$(ls /scratch/pmohseni/mert_emb_rp_synth/*.npy 2>/dev/null | wc -l)
echo "embeddings written: $N / 25"
[ "$N" -eq 25 ] || { echo "FATAL: incomplete bank"; exit 1; }
echo "Finished at $(date)"
