"""Render the demo cases: figures, audio, and a manifest for the page."""
import collections
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.musical_cases import (FPS, classify, load_piece,
                                               load_traj, merge_pages)
from extensions.analysis.render_cases import page_figure, path_figure, taxonomy_figure, tier_figure

OUT = sys.argv[1] if len(sys.argv) > 1 else '/scratch/pmohseni/omr/demo'
T = '/scratch/pmohseni/omr/traj'
CASES = [
    ('ChopinFF__O9__nocturne_in_b-flat_minor_room', 'biggest gain: 63.7 -> 77.3'),
    ('MussorgskyM__pictures-at-an-exhibition__promenade-3_room', 'hardest: 46.4, no gain'),
    ('BachJS__BWV797__bwv797_room', 'wrong-staff failures: 83.2 -> 87.3'),
    ('BachJS__BWVAnh116__anna-magdalena-07_room', 'clean success: 94.1 -> 98.2'),
]


def mp3(wav, dst):
    if os.path.exists(dst):
        return dst
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', wav,
                    '-ac', '1', '-ar', '22050', '-b:a', '64k', dst], check=True)
    return dst


def main():
    os.makedirs(OUT, exist_ok=True)
    base = load_traj(f'{T}/baseline_room.traj.npz')
    ours = load_traj(f'{T}/ours_room.traj.npz')
    tiers = {t: load_traj(f'{T}/ours_{t}.traj.npz') for t in ('room', 'do', 'rp_synth')}
    manifest, allcats = [], []

    for pn, note in CASES:
        short = pn.replace('_room', '')
        pc = load_piece(short, 'room')
        tb, to = merge_pages(base, pn), merge_pages(ours, pn)
        co = classify(to, pc)
        cb = classify(tb, pc)
        stem = short.split('__')[-1][:36]
        f1 = path_figure(pn, {'cyolo_sb baseline': tb, 'ours': to}, pc,
                         f'{OUT}/{stem}_path.png',
                         title=f'{short}   ·   {note}')
        # the page carrying the most error is the one worth showing
        pages = collections.Counter(to['page'][co['err'] > 0.5])
        pg = pages.most_common(1)[0][0] if pages else int(to['page'][0])
        f2 = page_figure(short, to, pc, int(pg), f'{OUT}/{stem}_page.png')
        per = {t: merge_pages(v, pn.replace('_room', f'_{t}') if t != 'room' else pn)
               for t, v in tiers.items()}
        per = {k: v for k, v in per.items() if v is not None}
        f3 = tier_figure(short, per, f'{OUT}/{stem}_tiers.png') if len(per) > 1 else None
        a = mp3(pc['wav'], f'{OUT}/{stem}.mp3')
        cc = collections.Counter(co['cat'])
        manifest.append(dict(
            piece=short, note=note, stem=stem,
            n=int(len(co['err'])),
            base=round(100 * float((cb['err'] <= .5).mean()), 1),
            ours=round(100 * float((co['err'] <= .5).mean()), 1),
            page=int(pg), audio=os.path.basename(a),
            figs=[os.path.basename(x) for x in (f1, f2, f3) if x],
            cats={k: int(v) for k, v in cc.items() if k != 'ok'}))
        print(f'  {short[:48]:<50} base {manifest[-1]["base"]:5.1f} -> '
              f'ours {manifest[-1]["ours"]:5.1f}   page {pg}')

    for pn in sorted({p.rsplit('_page_', 1)[0] for p in ours}):
        pc = load_piece(pn.replace('_room', ''), 'room')
        allcats.extend(classify(merge_pages(ours, pn), pc)['cat'])
    counts = {k: v for k, v in collections.Counter(allcats).items() if k != 'ok'}
    taxonomy_figure(counts, f'{OUT}/taxonomy.png')
    json.dump(dict(cases=manifest, taxonomy=counts,
                   total_frames=len(allcats)), open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(f'\ntaxonomy: {counts}')
    print(f'wrote {OUT}/manifest.json')


if __name__ == '__main__':
    main()
