"""C1 probe: does the aux loss actually attach, get a target with the right
shape, and produce gradient into the audio encoder? Throwaway."""
import os, sys, torch, numpy as np
_T=os.path.dirname(os.path.abspath(__file__)); _R=os.path.abspath(os.path.join(_T,'..','..'))
_CY=os.environ.get('CYOLO_ROOT','/scratch/pmohseni/datasets/cyolo_score_following')
sys.path.insert(0,_R); sys.path.insert(0,_CY)
D='/scratch/pmohseni/datasets/cyolo_data/msmd'; P='/scratch/pmohseni/amt_post_cyolo'
from extensions.hooks.cyolo_aux_patch import patch_cyolo_aux
patch_cyolo_aux({f'{D}/msmd_valid': f'{P}/msmd_valid'}, weight=1.0)
import cyolo_score_following.dataset as ds
import cyolo_score_following.utils.loss as lm
assert getattr(ds.SequenceDataset,'_c1_patched',False), 'dataset not patched'
assert getattr(lm,'_c1_patched',False), 'loss not patched'
assert getattr(ds,'_c1_iterate_patched',False), 'iterate not patched'
print('sentinels OK')
# head shape + gradient
from extensions.heads.posteriorgram_distill import PosteriorgramDistillHead, distill_loss
z=torch.randn(8,128,requires_grad=True)
h=PosteriorgramDistillHead(zdim=128,out_dim=176)
tgt=torch.rand(8,176)
loss,n=distill_loss(h(z),tgt)
loss.backward()
print('aux loss=%.4f n=%d  grad into z: norm=%.4f'%(loss,n,float(z.grad.norm())))
assert float(z.grad.norm())>0, 'no gradient reaches the conditioning vector'
# a real posteriorgram row is in [0,1] and BCE-compatible
import glob
f=sorted(glob.glob(f'{P}/msmd_valid/*.npy'))[0]
a=np.load(f).astype(np.float32)
print('real target %s range=[%.3f, %.3f]'%(a.shape,a.min(),a.max()))
assert a.shape[1]==176 and a.min()>=0 and a.max()<=1
print('ALL GOOD')
