#!/bin/bash
#SBATCH --job-name=mertroom
#SBATCH --account=def-ichiro
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/mertroom-%j.log
# MERT on the ROOM recordings, on CYOLO's own frame grid, so the MERT-audio
# detector (H1) can be run on the test set. Same script and settings as the
# clean training bank (precompute_mert_cyolo.sh without an IR bank): the room
# recording is the degradation here. Only *_room.wav -- msmd_rp also holds the
# direct-pickup and synthetic takes of the same performances.
set -uo pipefail
echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source /project/def-ichiro/pmohseni/music-alignment/.venv/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export HF_HOME=/scratch/pmohseni/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
timeout 7200 python -c "import torch,transformers,librosa,soundfile;print('imports ok')" \
    || { echo "FATAL: venv imports failed or stalled"; exit 1; }
DATA=/scratch/pmohseni/datasets/cyolo_data/msmd
W=/scratch/pmohseni/mert_room_wavs; mkdir -p "$W"
for f in "$DATA"/msmd_rp/*_room.wav; do ln -sf "$f" "$W/"; done
echo "room wavs: $(ls "$W"/*.wav | wc -l)"
OUT=/scratch/pmohseni/mert_emb_cyolo/msmd_rp_room
python -m scripts.precompute_mert_cyolo --wav_dir "$W" --out_dir "$OUT"
echo "bank: $(ls "$OUT"/*.npy | wc -l) npy"
echo "Job finished at $(date)"
