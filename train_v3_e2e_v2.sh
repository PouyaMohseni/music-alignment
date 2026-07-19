#!/bin/bash
#SBATCH --job-name=v3e2e-v2
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v3_e2e_v2/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v3_e2e_v2/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v3_e2e_v2

source .venv/bin/activate
export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"

# Clean continuation of the v3_e2e lineage, into its own results dir so the
# results/v3_e2e vs results/v3_e2e_long ambiguity (short first run reported as
# final; results/v3_e2e_long never independently evaluated) doesn't repeat.
#
# Resumes from results/v3_e2e_long/checkpoint_002000.pt, the most-trained
# surviving checkpoint from that lineage (2000 steps, i.e. checkpoint_000500
# from v3_e2e plus 1500 further e2e steps under v3_e2e_long).
#
# checkpoint_002000.pt's trainable_state already contains BOTH the e2e-tuned
# head and the LoRA encoder (train.py's _save() dumps every trainable param,
# not just the encoder), so init_encoder_checkpoint alone fully restores that
# state. We explicitly null out init_head_checkpoint: configs/v3_e2e.yaml's
# default (results/v3_all/checkpoint_002000.pt, a DIFFERENT, older,
# non-e2e-tuned head) would otherwise be loaded second in train.py's main()
# and silently clobber the e2e-trained head with that stale snapshot -- the
# same latent bug present (unexercised, since it happens to look like a no-op
# there) in train_v3_e2e_long.sh's invocation.
#
# train.py now tracks best-val-loss checkpoints (best_model.pt) and stops
# early after `early_stop_patience` (default 10) consecutive non-improving
# evals, so we can afford a generous step budget -- the run will end on its
# own once it plateaus rather than needing a hand-picked short step count.
python -m mymodel.v3_e2e.train \
  --config configs/v3_e2e.yaml \
  data.processed_root=/lustre07/scratch/pmohseni/music-alignment/data/MSMD/processed_all \
  data.num_workers=0 \
  train.init_encoder_checkpoint=results/v3_e2e_long/checkpoint_002000.pt \
  train.init_head_checkpoint=null \
  train.steps=8000 \
  optim.warmup_steps=100 \
  train.ckpt_every=500 \
  train.eval_every=500 \
  train.out_dir=results/v3_e2e_v2

echo "Job finished at $(date)"
