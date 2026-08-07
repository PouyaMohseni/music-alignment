"""Per-frame confidence + tracking-error dump for the CPJKU ConditionalUNet family.

WHY
---
Companion to scripts/dump_confidence_cyolo.py, for the *other* model family in
this project (CPJKU ConditionalUNet, Henkel et al. ISMIR 2020, and our MERT-based
descendants such as R3).  Score following has no published calibration
evaluation; the two existing "am I lost?" mechanisms are hand-built binary rules
(Brazier & Widmer EUSIPCO 2021 reliability factor; CODA ISMIR 2026 silence break
mode).  To score a continuous confidence signal against them we need, on the very
same frames, both the tracking error and every candidate confidence signal.

HOW
---
`patch_confidence_dump(out_path)` wraps
`audio_conditioned_unet.dataset.calculate_batch_stats` (the function
`iterate_dataset` calls once per BPTT chunk).  The wrapper sees the RAW
segmentation heatmap before the original binarises it in place, records the
signals below, then delegates to the original so the printed metric is bit-for-bit
what an unpatched eval would print.  An atexit hook writes the NPZ.

REQUIRES --batch_size 1 (every eval in this repo uses it, because strips have
variable width).  With bs=1 each piece's frames arrive strictly in temporal
order, which is what the frame counter and the Brazier slope rule rely on.

SIGNALS RECORDED PER FRAME
--------------------------
    err_frames  |interpol_c2o(x_pred) - interpol_c2o(x_gt)| in onset frames --
                the exact quantity eval_model.py turns into pct@0.5s
                (seconds = err_frames / fps, fps = 20).
    is_onset    whether the frame carries a note onset.  pct@0.5s is reported
                over onset frames only (--eval_onsets), so headline AUROCs use
                this subset.
    conf_max    max of the sigmoid heatmap.  The network's own peak activation:
                the closest analogue to YOLO objectness.
    conf_mass   mean of the heatmap x-marginal peak relative to total mass,
                i.e. p_max of the normalised x-marginal -- scale-free "how
                concentrated is the prediction".
    ent_x       Shannon entropy (nats) of the normalised x-marginal
                p(x) = sum_y heatmap[y, x] / total.
    margin      p(x) at the top mode minus p(x) at the best mode at least
                MIN_SEP_FRAC * W pixels away -- multimodality of the position
                posterior, i.e. "is there a competing place in the score".
    x_pred/x_gt unrolled x pixel of prediction / ground truth (strip pixels),
                for the Brazier & Widmer slope rule.
"""
import atexit
import os

import numpy as np
import torch


def patch_confidence_dump(out_path, min_sep_frac=0.05):
    import audio_conditioned_unet.dataset as ds
    from audio_conditioned_unet.utils import center_of_mass

    orig = ds.calculate_batch_stats
    rec = {}
    counter = {}

    def wrapped(pred, y_batch, piece_stats, current_pipeline, onsets, eval_center_of_mass,
                eval_only_onsets, threshold):
        raw = pred.detach().clone().view(y_batch.shape[0], y_batch.shape[1], *y_batch.shape[2:])
        binar = (raw >= threshold).float()

        sl, bs = y_batch.shape[0], y_batch.shape[1]
        W = y_batch.shape[-1]
        min_sep = max(1, int(min_sep_frac * W))

        for num, piece in enumerate(current_pipeline):
            fname = piece['file_name']
            if fname not in rec:
                rec[fname] = {k: [] for k in ('frame', 'is_onset', 'err_frames', 'conf_max',
                                              'conf_mass', 'ent_x', 'margin', 'x_pred', 'x_gt')}
                counter[fname] = 0

            staff_coords, add_per_staff = current_pipeline[num]['add_per_staff']
            c2o = current_pipeline[num]['interpol_c2o']

            for t in range(sl):
                hm = raw[t, num, 0]
                px = hm.sum(dim=0)
                tot = float(px.sum())
                if tot <= 0:
                    p = np.full(W, 1.0 / W)
                else:
                    p = (px / tot).cpu().numpy().astype(np.float64)

                k = int(np.argmax(p))
                nz = p[p > 0]
                ent = float(-(nz * np.log(nz)).sum())
                far = np.abs(np.arange(W) - k) >= min_sep
                second = float(p[far].max()) if far.any() else 0.0

                com_gt = center_of_mass(y_batch[t, num, 0])
                if binar[t, num, 0].sum() == 0:
                    com_pred = torch.zeros_like(com_gt)
                else:
                    com_pred = center_of_mass(binar[t, num, 0])
                cg = com_gt.cpu().numpy()
                cp = com_pred.cpu().numpy()

                sid_p = int(np.argwhere(min(staff_coords, key=lambda x: abs(x - cp[0])) == staff_coords).item())
                sid_g = int(np.argwhere(min(staff_coords, key=lambda x: abs(x - cg[0])) == staff_coords).item())
                xp = cp[1] + add_per_staff[sid_p]
                xg = cg[1] + add_per_staff[sid_g]

                r = rec[fname]
                r['frame'].append(counter[fname])
                r['is_onset'].append(bool(onsets[num][t]) if onsets else False)
                r['err_frames'].append(float(abs(c2o(xp) - c2o(xg))))
                r['conf_max'].append(float(hm.max()))
                r['conf_mass'].append(float(p[k]))
                r['ent_x'].append(ent)
                r['margin'].append(float(p[k] - second))
                r['x_pred'].append(float(xp))
                r['x_gt'].append(float(xg))
                counter[fname] += 1

        return orig(pred, y_batch, piece_stats, current_pipeline, onsets, eval_center_of_mass,
                    eval_only_onsets, threshold)

    ds.calculate_batch_stats = wrapped

    def _write():
        if not rec:
            print('[confidence_dump] nothing recorded')
            return
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        flat = {}
        for fname, r in rec.items():
            for k, v in r.items():
                flat[f'{fname}||{k}'] = np.asarray(v)
        flat['__fps__'] = np.asarray([20.0])
        np.savez_compressed(out_path, **flat)
        n = sum(len(r['err_frames']) for r in rec.values())
        print(f'[confidence_dump] wrote {out_path}: {len(rec)} pages, {n} frames')

    atexit.register(_write)
    print(f'[confidence_dump] patched calculate_batch_stats -> {out_path}')
