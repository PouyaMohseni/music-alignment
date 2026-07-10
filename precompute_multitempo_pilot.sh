#!/bin/bash
#SBATCH --job-name=multitempo-pilot
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/multitempo_pilot-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/multitempo_pilot-%j.log

# E4 PILOT -- scoped-down first test before committing the full-scale
# ~10-20 GPU-hour precompute (precompute_multitempo_wholepiece.sh, not yet
# run). 100 train pieces x ONE new tempo factor (750) instead of 354 x 2 --
# ~14x cheaper, enough to get a real (if noisy) read on whether multi-tempo
# training helps D2 at all before committing more compute. Cluster is
# already running 13 other jobs; this is deliberately modest.

echo "Job started on $(hostname) at $(date)"
cd /project/def-ichiro/pmohseni/music-alignment
source .venv/bin/activate
export TRANSFORMERS_OFFLINE=1

RENDER_DIR=/scratch/pmohseni/cpjku_fmt_multitempo
MERT_OUT=/scratch/pmohseni/mert_emb_zenodo/cpjku_fmt_wholepiece_multitempo
mkdir -p "$RENDER_DIR" "$MERT_OUT"

echo "=== Step 1: render tempo_750 for first 100 train pieces ==="
python3 -c "
import yaml
train = yaml.safe_load(open('data/MSMD/cpjku_fmt/split_train.yaml'))['files']
with open('/tmp/pilot_pieces.txt', 'w') as f:
    for p in train[:100]:
        f.write(p + '\n')
"
python scripts/render_multitempo_wholepiece.py \
    --performance_dir data/MSMD/cpjku_fmt/performance \
    --out_dir "$RENDER_DIR" \
    --tempo_factors 750 \
    --pieces_file /tmp/pilot_pieces.txt \
    --sound_font third_party/cpjku_unet/audio_conditioned_unet/sound_fonts/grand-piano-YDP-20160804.sf2 \
    --fluidsynth /scratch/pmohseni/micromamba/envs/fluidsynth/bin/fluidsynth

echo "=== Step 2: MERT-encode the new renders (GPU) ==="
python3 -c "
import yaml, glob, os
names = sorted(set(os.path.basename(f)[:-4] for f in glob.glob('$RENDER_DIR/*.wav')))
yaml.dump({'files': names}, open('/tmp/pilot_multitempo_split.yaml', 'w'))
print(len(names), 'tempo-variant files to encode')
"
python scripts/precompute_mert_test_eval.py \
    --wav_dir "$RENDER_DIR" \
    --split_file /tmp/pilot_multitempo_split.yaml \
    --out_dir "$MERT_OUT"

echo "Job finished at $(date)"
