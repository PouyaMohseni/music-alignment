#!/bin/bash
#SBATCH --job-name=multitempo-wholepiece
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/multitempo_wholepiece-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/multitempo_wholepiece-%j.log

# E4 prerequisite, FULL SCALE -- NOT YET RUN (see this run's report for cost
# measurement: ~8-10 min/piece per tempo factor on CPU for MERT encoding
# alone, i.e. 354 train pieces x 2 new tempo factors x ~9min = ~106 CPU-hours;
# this MUST run on GPU like every other MERT precompute in this project, but
# even at a plausible 5-10x GPU speedup that's still 10-20 wall-clock hours --
# review this estimate before submitting, may need chunking across multiple
# jobs or trimming to fewer tempo factors / a train-set subset.
#
# Step 1: render tempo-scaled whole-piece MIDI+WAV (CPU-bound, fluidsynth).
# Step 2: MERT-encode the new renders (GPU-bound).

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

RENDER_DIR=/scratch/pmohseni/cpjku_fmt_multitempo
MERT_OUT=/scratch/pmohseni/mert_emb_zenodo/cpjku_fmt_wholepiece_multitempo
mkdir -p "$RENDER_DIR" "$MERT_OUT"

echo "=== Step 1: render tempo_750 + tempo_1250 for all train+val pieces ==="
python3 -c "
import yaml
train = yaml.safe_load(open('data/MSMD/cpjku_fmt/split_train.yaml'))['files']
val = yaml.safe_load(open('data/MSMD/cpjku_fmt/split_val.yaml'))['files']
with open('/tmp/trainval_pieces.txt', 'w') as f:
    for p in train + val:
        f.write(p + '\n')
"
python scripts/render_multitempo_wholepiece.py \
    --performance_dir data/MSMD/cpjku_fmt/performance \
    --out_dir "$RENDER_DIR" \
    --tempo_factors 750 1250 \
    --pieces_file /tmp/trainval_pieces.txt \
    --sound_font third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2 \
    --fluidsynth /scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth

echo "=== Step 2: MERT-encode the new renders ==="
python3 -c "
import yaml, glob, os
names = sorted(set(os.path.basename(f)[:-4] for f in glob.glob('$RENDER_DIR/*.wav')))
yaml.dump({'files': names}, open('/tmp/multitempo_split.yaml', 'w'))
print(len(names), 'tempo-variant files to encode')
"
python scripts/precompute_mert_test_eval.py \
    --wav_dir "$RENDER_DIR" \
    --split_file /tmp/multitempo_split.yaml \
    --out_dir "$MERT_OUT"

echo "Job finished at $(date)"
