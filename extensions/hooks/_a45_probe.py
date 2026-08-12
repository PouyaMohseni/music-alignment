"""Throwaway probe: do the A4 and A5 patch chains work under the REAL sbatch
invocation (cwd inside the cpjku package, entry point by absolute path)?
Exercises the patches and one synthetic forward/backward through the heads,
including the optimizer registration. Safe to delete."""
import os, sys, torch
_T = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_T, '..', '..'))
_C = os.path.join(_R, 'third_party', 'cpjku_unet')
sys.path.insert(0, _R); sys.path.insert(0, _C)
print('cwd:', os.getcwd())

# ---- A5: FiLM class swap must happen before construction ----------------
from extensions.heads.adaln_zero import patch_adaln_zero, AdaLNZero
patch_adaln_zero()
from audio_conditioned_unet import network as nm
assert issubclass(nm.FiLM, AdaLNZero), 'A5 patch did not take'
print('A5 OK: network.FiLM is AdaLNZero')

# ---- A4: iterate_dataset swap + heads train end-to-end ------------------
from extensions.hooks.boundary_patch import patch_boundary, _staff_rows_from_logits
patch_boundary()
import audio_conditioned_unet.dataset as ds
fn = getattr(ds, 'iterate_dataset')
name = getattr(getattr(fn, 'func', fn), '__name__', '?')
assert name == 'iterate_dataset_boundary', name
print('A4 OK: iterate_dataset ->', name)

from extensions.heads.boundary_head import BoundaryHead, boundary_loss, decode_mask
from extensions.heads.staff_coarse_head import StaffCoarseHead, staff_loss
N, C, H, W = 4, 24, 40, 100
feat = torch.randn(N, C, H // 2, W // 2, requires_grad=True)
y = torch.zeros(N, 1, H, W)
for i in range(N):
    y[i, 0, 8 + 6 * i, 20 + 15 * i] = 1.0
    y[i, 0, 7 + 6 * i:10 + 6 * i, 18 + 15 * i:23 + 15 * i] = 1.0

bh, sh = BoundaryHead(in_ch=C), StaffCoarseHead(in_ch=C, n_bins=16)
opt = torch.optim.Adam(list(bh.parameters()) + list(sh.parameters()), lr=1e-3)
a, dl, dr = bh(feat); s = sh(feat)
y_small = torch.nn.functional.adaptive_avg_pool2d(y.view(-1, H, W).unsqueeze(1), (H, a.shape[1]))
l1, parts = boundary_loss(a, dl, dr, y_small)
l2, _ = staff_loss(s, y, 16)
loss = l1 + l2
print('loss=%.4f  parts=%s  staff=%.4f' % (loss, {k: round(v,4) for k,v in parts.items() if k!='n'}, l2))
loss.backward()
gn = sum(float(p.grad.norm()) for p in bh.parameters() if p.grad is not None)
assert gn > 0, 'no gradient reached the boundary head'
print('grad norm through BoundaryHead = %.4f' % gn)
opt.step()

# decode -> mask -> the real metric path
rows = _staff_rows_from_logits(s.detach(), H)
scale = W / a.shape[1]
m = decode_mask(a.detach(), dl.detach()*scale, dr.detach()*scale, H, staff_row=rows)
if m.shape[-1] != W:
    m = torch.nn.functional.interpolate(m, size=(H, W), mode='nearest')
print('decoded mask %s  survivors/frame=%s' % (tuple(m.shape), [int(v) for v in (m>=0.5).sum(dim=(1,2,3))]))
assert m.shape == (N,1,H,W) and int((m>=0.5).sum()) > 0, 'decode empty -> every frame a miss'
print('ALL GOOD')
