"""H2 probe: does the patch read the posteriorgram bank's 176-dim width, and
does feature-space augmentation run without a device/shape fault? Throwaway."""
import os, sys, numpy as np, torch
_T=os.path.dirname(os.path.abspath(__file__)); _R=os.path.abspath(os.path.join(_T,'..','..'))
_CY=os.environ.get('CYOLO_ROOT','/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0,_R); sys.path.insert(0,_CY)
D='/scratch/pmohseni/datasets/cyolo_data/msmd'
P='/scratch/pmohseni/amt_post_cyolo'
from extensions.hooks import cyolo_mert_patch as cmp
n=cmp.patch_cyolo_mert({f'{D}/msmd_valid': f'{P}/msmd_valid'}, feat_aug=1.0)
assert cmp.MERT_DIM==176, f'dim read as {cmp.MERT_DIM}, expected 176'
print('dim OK:', cmp.MERT_DIM, ' bank entries:', n)
from cyolo_score_following.models.yolo import Model
T=120
flat=torch.randn(T*176)
out=Model.compute_spec(None,[flat],tempo_aug=True)
e=out[0]
print('feat_aug output %s finite=%s  (input was %d frames)'%(tuple(e.shape), bool(torch.isfinite(e).all()), T))
assert e.shape[1]==176 and bool(torch.isfinite(e).all())
out2=Model.compute_spec(None,[flat],tempo_aug=False)
print('no-aug output   %s  (must be exactly %d frames)'%(tuple(out2[0].shape), T))
assert out2[0].shape==(T,176), 'eval path must not resample or mask'
print('ALL GOOD')
