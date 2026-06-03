#!/bin/bash
#SBATCH --job-name=music-align-nce
#SBATCH --account=def-ichiro
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --tmp=64G
#SBATCH --output=/project/def-ichiro/pmohseni/music-alignment/results/v1_nce/slurm-%j.log
#SBATCH --error=/project/def-ichiro/pmohseni/music-alignment/results/v1_nce/slurm-%j.log

echo "Job started on $(hostname) at $(date)"
nvidia-smi

cd /project/def-ichiro/pmohseni/music-alignment
mkdir -p results/v1_nce

source .venv/bin/activate
pip install -q peft

# Copy dataset to local NVMe SSD
echo "==> copying processed_all to \$SLURM_TMPDIR via tar pipe..."
mkdir -p $SLURM_TMPDIR/processed_all
time tar cf - -C /lustre07/scratch/pmohseni/music-alignment/data/MSMD processed_all \
  | tar xf - -C $SLURM_TMPDIR/
echo "==> dataset copy done, $(ls $SLURM_TMPDIR/processed_all | wc -l) dirs"

export HF_HOME=/project/def-ichiro/pmohseni/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"
echo "peft:  $(python -c 'import peft; print(peft.__version__)')"

python -m mymodel.v1_baseline.train \
  --config configs/v1_lora.yaml \
  train.steps=30000 \
  train.batch_size=4 \
  train.grad_accum_steps=4 \
  data.num_workers=4 \
  loss.nce_gate_threshold=999.0 \
  train.out_dir=results/v1_nce \
  data.manifest_path=$SLURM_TMPDIR/processed_all/manifest.jsonl

echo "Job finished at $(date)"
