"""Quick smoke test: run M01's actual train() function for a handful of
epochs on a small piece subset with batch_size=8, confirm loss decreases
under the exact production code path (not a standalone reimplementation).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from mymodel.cadp.m01_train import train

cfg = OmegaConf.load('configs/cadp_m01.yaml')
cfg.train.max_epochs = 15
cfg.train.batch_size = 8
cfg.train.out_dir = '/scratch/pmohseni/results/cadp_m01_smoketest'
cfg.optim.patience = 100  # disable early stop for this quick check
train(cfg)
