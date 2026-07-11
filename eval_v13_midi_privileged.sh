#!/bin/bash
#SBATCH --job-name=eval-v13-midi
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_midi-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/eval_v13_midi-%j.log

# E2/E3 interim eval: v13's OWN eval.py, unmodified -- the MIDI-privileged
# checkpoint's state_dict is byte-identical to v13's original format
# (MidiEncoder is a separate module, never attached to `network`), so this
# needs zero special-casing. Checkpoint is from the FIRST training attempt
# (timed out at 24h, epoch ~5-ish last logged) -- interim read, not final.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
module load gcc opencv
source .venv/bin/activate
export OMP_NUM_THREADS=4
export TRANSFORMERS_OFFLINE=1

python -m mymodel.v13_mert_unet.eval \
    --checkpoint /scratch/pmohseni/results/v13_midi_privileged/best_model.pt \
    --config     configs/v13_midi_privileged.yaml \
    --split      test \
    --out_dir    /scratch/pmohseni/results/v13_midi_privileged/eval

echo "Job finished at $(date)"
