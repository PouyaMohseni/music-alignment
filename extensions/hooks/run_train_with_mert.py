"""B1 entry point: applies the MERT monkey-patch, then runs CPJKU's own
unmodified train_model.py in-process via runpy (so its
`if __name__ == '__main__':` block executes exactly as if invoked directly,
argparse reading the real sys.argv this script is called with).

Requires MERT_PATH_MAP env var: semicolon-separated dataset_path=emb_root
pairs, one per dataset path passed via --train_set/--val_set (train_model.py
calls load_dataset twice, once per split, each needing its own precomputed-
MERT directory). Example:
    MERT_PATH_MAP="/scratch/pmohseni/msmd_train_full=/scratch/pmohseni/mert_emb_zenodo/train_full;../data/msmd/msmd_valid=/scratch/pmohseni/mert_emb_zenodo/msmd_valid" \
    python extensions/hooks/run_train_with_mert.py \
        --film_layers 2 3 4 5 6 7 8 --use_lstm --augment \
        --train_set /scratch/pmohseni/msmd_train_full --val_set ../data/msmd/msmd_valid \
        --config ... --audio_encoder MERTProjector --tag B1a_mert_swap
"""
import os
import runpy
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_THIS_DIR, '..', '..')
# audio_conditioned_unet is pip-installed editable in this venv (venv_cpjku310),
# so it's importable regardless of cwd/sys.path -- only `extensions` needs adding.
_CPJKU_PKG_DIR = os.path.join(_PROJECT_ROOT, 'third_party', 'cpjku_unet', 'audio_conditioned_unet')

sys.path.insert(0, _PROJECT_ROOT)     # for `extensions` package

from extensions.hooks.mert_patch import patch_mert_pipeline

path_map_str = os.environ.get('MERT_PATH_MAP')
if not path_map_str:
    raise RuntimeError('MERT_PATH_MAP env var must be set: '
                        '"dataset_path1=emb_root1;dataset_path2=emb_root2"')
path_to_emb_root = dict(pair.split('=', 1) for pair in path_map_str.split(';') if pair)
patch_mert_pipeline(path_to_emb_root=path_to_emb_root)

runpy.run_path(os.path.join(_CPJKU_PKG_DIR, 'train_model.py'), run_name='__main__')
