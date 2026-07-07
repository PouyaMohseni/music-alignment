#!/bin/bash
#SBATCH --job-name=c4-tempo-contrastive
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/c4_tempo_contrastive-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/c4_tempo_contrastive-%j.log

# C4: self-supervised tempo-invariant contrastive pretraining of CBEncoder,
# exploiting MSMD's own multi-tempo renders of the same piece as free
# positive pairs (see extensions/pretrain/tempo_contrastive.py for the
# method and the verification that note order/count match exactly across
# tempo factors). Standalone -- pretrains CBEncoder in isolation, not the
# full ConditionalUNet; NOT yet integrated into a real CB_TA training run
# (see run_pretrain_c4.py's docstring for the intended follow-up warm-start
# usage). Runs in the main .venv (needs torch/librosa/pretty_midi, none of
# which venv_cpjku310 has) with fluidsynth invoked via absolute path.

set -euo pipefail
echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate

OUT=/scratch/pmohseni/results/c4_tempo_contrastive
mkdir -p "$OUT"

python extensions/pretrain/run_pretrain_c4.py \
    --train_dir /scratch/pmohseni/msmd_train_full \
    --sound_font third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2 \
    --fluidsynth /scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth \
    --out_dir "$OUT" \
    --spec_enc 32 \
    --batch_size 32 \
    --steps 20000 \
    --lr 1e-3 \
    --temperature 0.1 \
    --save_every 1000

echo ""
echo "Training finished at $(date)"
echo "Pretrained encoder: $OUT/c4_encoder_latest.pt"
