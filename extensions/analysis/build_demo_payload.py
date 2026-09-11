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
from extensions.analysis.musical_cases import (FPS, classify, load_piece, load_traj,
                                               merge_pages, summarize)

T = '/scratch/pmohseni/omr/traj'
OUT = os.environ.get('DEMO_OUT', '/scratch/pmohseni/omr/demo')
TITLES = {
    'BachJS__BWV797__bwv797_room': 'Bach, Sinfonia 11 BWV 797',
    'BachJS__BWV117a__BWV-117a_room': 'Bach, BWV 117a',
    'BachJS__BWV830__BWV-830-2_room': 'Bach, Partita BWV 830',
    'BachJS__BWVAnh120__BWV-120_room': 'Bach, BWV 120',
    'MozartWA__KV331__KV331_1_2_var1_room': 'Mozart, KV 331 Var. 1',
    'BachJS__BWVAnh113__anna-magdalena-03_room': 'Bach, Anna Magdalena 3',
    'BachJS__BWVAnh116__anna-magdalena-07_room': 'Bach, Anna Magdalena 7',
    'BachJS__BWV924a__bach-prelude-bwv924a_room': 'Bach, Prelude BWV 924a',
    'ChopinFF__O9__nocturne_in_b-flat_minor_room': 'Chopin, Nocturne Op. 9 No. 1',
    'BachJS__BWV817__bach-french-suite-6-menuet_room': 'Bach, French Suite 6 Menuet',
    'MussorgskyM__pictures-at-an-exhibition__promenade-3_room': 'Mussorgsky, Promenade 3',
    'SchumannR__O68__schumann-op68-01-melodie_room': 'Schumann, Melodie Op. 68 No. 1',
    'SchumannR__O68__schumann-op68-06-pauvre-orpheline_room':
        'Schumann, Pauvre Orpheline Op. 68 No. 6',
    'SchumannR__O68__schumann-op68-08-cavalier-sauvage_room':
        'Schumann, Cavalier Sauvage Op. 68 No. 8',
    'SchumannR__O68__schumann-op68-16-premier-chagrin_room':
        'Schumann, Premier Chagrin Op. 68 No. 16',
    'SchumannR__O68__schumann-op68-26-sans-titre_room': 'Schumann, Sans Titre Op. 68 No. 26',
}


def acc(traj, pn):
    t = merge_pages(traj, pn)
    return 100.0 * float((np.abs(t['t_pred'] - t['t_gt']) / FPS <= .5).mean())


def pick_cases(base, flat, ours):
    """One case per distinct behaviour, chosen by RULE from the trajectories.

    The list used to be typed in from a per-piece table, with its numbers in
    the captions, and went stale every time the shipped checkpoint changed
    (it still quoted the 91.4 model's numbers after vel_p8 replaced it). Now
    the four roles are fixed and the pieces filling them are recomputed:

      the biggest gain over the baseline, the second biggest, the worst piece
      after decoding, and the piece where the image features cost most
      against the featureless selector (or, if they cost nothing anywhere,
      where they help most).
    """
    names = sorted({p.rsplit('_page_', 1)[0] for p in ours})
    b = {n: acc(base, n) for n in names}
    f = {n: acc(flat, n) for n in names}
    o = {n: acc(ours, n) for n in names}
    print(f'{"piece":52s} {"base":>6s} {"flat":>6s} {"ours":>6s}')
    for n in sorted(names, key=lambda n: b[n] - o[n]):
        print(f'{n[:52]:52s} {b[n]:6.1f} {f[n]:6.1f} {o[n]:6.1f}')
    g1, g2 = sorted(names, key=lambda n: o[n] - b[n], reverse=True)[:2]
    rest = [n for n in names if n not in (g1, g2)]
    w = min(rest, key=lambda n: o[n])
    rest = [n for n in rest if n != w]
    c = max(rest, key=lambda n: f[n] - o[n])
    if f[c] - o[c] > 0.5:
        why4 = f'image features cost us here: featureless {f[c]:.1f}, ours {o[c]:.1f}'
    else:
        c = max(rest, key=lambda n: o[n] - f[n])
        why4 = f'where image features help most: featureless {f[c]:.1f}, ours {o[c]:.1f}'
    near = ', and near-perfect after' if o[g2] >= 97 else ''
    worse = ', and we make it worse' if o[w] < b[w] else ''
    return [
        (g1, TITLES.get(g1, g1), f'biggest gain in the set: {b[g1]:.1f} to {o[g1]:.1f}'),
        (g2, TITLES.get(g2, g2), f'second biggest{near}: {b[g2]:.1f} to {o[g2]:.1f}'),
        (w, TITLES.get(w, w), f'worst piece{worse}: {b[w]:.1f} to {o[w]:.1f}'),
        (c, TITLES.get(c, c), why4),
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
    # the shipped model: vel_p8 at 93.4, not the featureless 91.4 selector the
    # demo was first built from. Overridable so the panels can be regenerated
    # for a different checkpoint without editing this file.
    ours = load_traj(os.environ.get('OURS_TRAJ', f'{T}/velp8_room.traj.npz'))
    # the featureless selector at the same blend, for the fourth role
    flat = load_traj(os.environ.get('FLAT_TRAJ', f'{T}/selected_room.traj.npz'))
    os.makedirs(OUT, exist_ok=True)
    cases = []
    for pn, title, why in pick_cases(base, flat, ours):
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
    # failure taxonomy over EVERY scored onset of the model the panels show;
    # the page's table was last computed for the 91.4 featureless model
    allcats = []
    for pn in sorted({p.rsplit('_page_', 1)[0] for p in ours}):
        pc = load_piece(pn.replace('_room', ''), 'room')
        allcats.extend(classify(merge_pages(ours, pn), pc)['cat'])
    n, bad, share = summarize(allcats)
    print(f'\ntaxonomy: {n} onsets, {bad} outside threshold ({100.0 * (n - bad) / n:.2f}%)')
    for k, (v, pct) in sorted(share.items(), key=lambda t: -t[1][0]):
        print(f'  {k:14s} {v:5d}  {pct:5.1f}%')
    payload = dict(cases=cases)
    p = f'{OUT}/payload.json'
    json.dump(payload, open(p, 'w'))
    print(f'\nwrote {p}  ({os.path.getsize(p) / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
