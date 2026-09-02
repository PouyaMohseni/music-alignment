"""Self-contained payload for the demo page: score image, audio, and the path.

Everything is cut to ONE page of the score, because that is the unit a viewer
can actually follow: the audio excerpt covers exactly the span the tracker spent
on that page, and the trajectory is the frames recorded over that span. A marker
driven by the audio's own currentTime then shows where the system thinks it is,
against where it should be, on the real score.
"""
import base64
import io
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '/project/def-ichiro/pmohseni/music-alignment')
from extensions.analysis.musical_cases import FPS, classify, load_piece, load_traj, merge_pages

T = '/scratch/pmohseni/omr/traj'
OUT = '/scratch/pmohseni/omr/demo'
# Chosen from the per-piece table for the SHIPPED 91.4 model, one case per
# distinct behaviour rather than four wins:
#
#   piece                         base   hand   ours   delta
#   Chopin Nocturne Op.9          63.7   77.3   87.5  +23.8   biggest gain
#   Schumann op.68 no.1           86.3   91.4   99.4  +13.1   near-perfect result
#   Mussorgsky Promenade 3        46.4   46.4   39.3   -7.1   worst, and we hurt it
#   Schumann op.68 no.16          85.4   92.3   84.6   -0.8   hand decode beats ours
CASES = [
    ('ChopinFF__O9__nocturne_in_b-flat_minor_room', 'Chopin, Nocturne Op. 9 No. 1',
     'biggest gain in the set: 63.7 to 87.5'),
    ('SchumannR__O68__schumann-op68-01-melodie_room', 'Schumann, Melodie Op. 68 No. 1',
     'near-perfect: 86.3 to 99.4'),
    ('MussorgskyM__pictures-at-an-exhibition__promenade-3_room',
     'Mussorgsky, Promenade 3', 'worst piece, and we make it worse: 46.4 to 39.3'),
    ('SchumannR__O68__schumann-op68-16-premier-chagrin_room',
     'Schumann, Premier Chagrin Op. 68 No. 16',
     'the learned selector loses here: hand decode 92.3, ours 84.6'),
]


def b64(path, mime):
    with open(path, 'rb') as f:
        return f'data:{mime};base64,' + base64.b64encode(f.read()).decode()


def png_b64(arr, max_w=1000):
    im = Image.fromarray(arr).convert('L')
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return ('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode(),
            im.width, im.height)


def main():
    base = load_traj(f'{T}/baseline_room.traj.npz')
    # the shipped 91.4 model, not the 86.5 hand decoder the demo first showed
    ours = load_traj(f'{T}/selected_room.traj.npz')
    cases = []
    for pn, title, why in CASES:
        short = pn.replace('_room', '')
        pc = load_piece(short, 'room')
        tb, to = merge_pages(base, pn), merge_pages(ours, pn)
        co, cb = classify(to, pc), classify(tb, pc)
        # show the page holding the most error -- that is where there is
        # something to look at
        errpg = to['page'][co['err'] > 0.5]
        pg = int(np.bincount(errpg).argmax()) if errpg.size else int(to['page'][0])
        m = to['page'] == pg
        mb = tb['page'] == pg
        t0, t1 = float(to['frame'][m][0] / FPS), float(to['frame'][m][-1] / FPS)
        stem = short.split('__')[-1][:32]
        wav, mp3 = pc['wav'], f'{OUT}/{stem}_p{pg}.mp3'
        if not os.path.exists(mp3):
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(max(t0 - .5, 0)),
                            '-to', str(t1 + .5), '-i', wav, '-ac', '1', '-ar', '22050',
                            '-b:a', '56k', mp3], check=True)
        sheet = pc['sheets'][pg]
        img, w, h = png_b64(sheet, 1000)
        sc = w / sheet.shape[1]
        pad = pc['pad']     # the loader pads pages to a square and shifts x
        cases.append(dict(
            id=stem, title=title, why=why, page=pg, img=img, w=w, h=h,
            audio=b64(mp3, 'audio/mpeg'), t0=max(t0 - .5, 0),
            base=round(100 * float((cb['err'] <= .5).mean()), 1),
            ours=round(100 * float((co['err'] <= .5).mean()), 1),
            page_ours=round(100 * float((co['err'][m] <= .5).mean()), 1),
            page_base=round(100 * float((cb['err'][mb] <= .5).mean()), 1),
            t=[round(float(v), 3) for v in to['frame'][m] / FPS],
            px=[round(float(v) * sc, 1) for v in to['x_pred'][m] - pad],
            py=[round(float(v) * sc, 1) for v in to['y_pred'][m]],
            gx=[round(float(v) * sc, 1) for v in to['x_gt'][m] - pad],
            gy=[round(float(v) * sc, 1) for v in to['y_gt'][m]],
            bx=[round(float(v) * sc, 1) for v in tb['x_pred'][mb] - pad],
            by=[round(float(v) * sc, 1) for v in tb['y_pred'][mb]],
            bt=[round(float(v), 3) for v in tb['frame'][mb] / FPS],
            e=[round(float(v), 3) for v in co['err'][m]],
            be=[round(float(v), 3) for v in cb['err'][mb]]))
        print(f'  {stem:<34} page {pg}  {t1 - t0:5.1f}s  '
              f'{len(cases[-1]["t"]):>4} frames  page acc {cases[-1]["page_ours"]:.1f}%')
    payload = dict(cases=cases)
    p = f'{OUT}/payload.json'
    json.dump(payload, open(p, 'w'))
    print(f'\nwrote {p}  ({os.path.getsize(p) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
