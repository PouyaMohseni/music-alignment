"""The three image-feature arms on CYOLO-SB candidates, under the LOPO
protocol of lopo_new, so the 90.6/90.9/90.9 claim rests on one measurement.

nofeat_ctrl was never in lopo_new's job list, so the "no image feature" number
in the paper predates the current protocol and is the one this checks.
"""
import sys
sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
import extensions.analysis.lopo_new as L

L.JOBS = [
    ('CYOLO-SB full',           f'{L.M}/nbr/nbrp64_s[0-5].pt', f'{L.O}/candf256/room.npz', None),
    ('CYOLO-SB no image feats', f'{L.M}/nofeat_ctrl.pt',       f'{L.O}/candf256/room.npz', None),
    ('CYOLO-SB DINOv2 feats',   f'{L.M}/abl/dino_s*.pt',       f'{L.O}/candf_dino/room.npz', None),
]
L.main()
