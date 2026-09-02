import json, os, sys
OUT = '/tmp/claude-3152518/-lustre06-project-6002780-pmohseni-music-alignment/221a9f7b-1956-4881-9c3f-4fbd1f392674/scratchpad'
pay = open('/scratch/pmohseni/omr/demo/payload.json').read()

HEAD = r'''<title>Following the Score</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#FBFAF7; --panel:#FFFFFF; --ink:#16151A; --ink-2:#4A4750;
  --rule:#D8D4CB; --rule-2:#EDEAE3;
  --track:#1B6B8C; --track-soft:#1B6B8C22;
  --miss:#B0343C; --truth:#7A7580;
  --shadow:0 1px 2px #16151A0F, 0 8px 24px #16151A0A;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#141319; --panel:#1B1A21; --ink:#E9E6DF; --ink-2:#A5A1AC;
  --rule:#35333C; --rule-2:#26252D;
  --track:#5AB6DA; --track-soft:#5AB6DA26;
  --miss:#E4707A; --truth:#8D8996;
  --shadow:0 1px 2px #0008, 0 8px 24px #0006;
}}
:root[data-theme="dark"]{
  --paper:#141319; --panel:#1B1A21; --ink:#E9E6DF; --ink-2:#A5A1AC;
  --rule:#35333C; --rule-2:#26252D;
  --track:#5AB6DA; --track-soft:#5AB6DA26;
  --miss:#E4707A; --truth:#8D8996;
  --shadow:0 1px 2px #0008, 0 8px 24px #0006;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 16px/1.6 "Source Sans 3",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:48px 24px 96px;
  display:flex;flex-direction:column;gap:44px}
h1{font:600 34px/1.15 Spectral,Georgia,serif;margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font:600 21px/1.25 Spectral,Georgia,serif;margin:0;text-wrap:balance}
.lede{color:var(--ink-2);max-width:64ch;margin:0}
.eyebrow{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;
  text-transform:uppercase;color:var(--ink-2)}
header{display:flex;flex-direction:column;gap:14px;
  border-bottom:1px solid var(--rule);padding-bottom:30px}
.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
  background:var(--rule-2);border:1px solid var(--rule-2);border-radius:3px;overflow:hidden}
.fig{background:var(--panel);padding:15px 17px;display:flex;flex-direction:column;gap:3px}
.fig b{font:500 27px/1 "IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;
  color:var(--track)}
.fig b.plain{color:var(--ink)}
.fig span{font-size:12.5px;color:var(--ink-2);line-height:1.35}
section{display:flex;flex-direction:column;gap:16px}
.case{background:var(--panel);border:1px solid var(--rule-2);border-radius:4px;
  box-shadow:var(--shadow);overflow:hidden}
.case-head{padding:16px 20px;border-bottom:1px solid var(--rule-2);
  display:flex;flex-wrap:wrap;gap:12px 20px;align-items:baseline}
.case-head h2{flex:1 1 260px}
.why{color:var(--ink-2);font-size:14px;flex:1 1 200px}
.score{position:relative;line-height:0;background:#fff;
  border-bottom:1px solid var(--rule-2);overflow-x:auto}
:root[data-theme="dark"] .score,
:root:not([data-theme="light"]) .score{filter:none}
.score img{width:100%;height:auto;display:block}
.score canvas{position:absolute;inset:0;width:100%;height:100%}
.transport{padding:14px 20px 18px;display:flex;flex-direction:column;gap:12px}
.row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
button{font:600 13.5px/1 "Source Sans 3",sans-serif;color:var(--paper);
  background:var(--track);border:0;border-radius:3px;padding:9px 17px;cursor:pointer;
  transition:opacity .12s}
button:hover{opacity:.85}
button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
button.ghost{background:transparent;color:var(--ink-2);border:1px solid var(--rule);
  padding:8px 13px}
button.ghost[aria-pressed="true"]{color:var(--track);border-color:var(--track);
  background:var(--track-soft)}
.clock{font:400 13px/1 "IBM Plex Mono",monospace;color:var(--ink-2);
  font-variant-numeric:tabular-nums}
.strip{position:relative;height:52px;width:100%;cursor:pointer;
  border:1px solid var(--rule-2);border-radius:3px;background:var(--paper)}
.strip canvas{position:absolute;inset:0;width:100%;height:100%;border-radius:3px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12.5px;color:var(--ink-2)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;
  margin-right:6px;vertical-align:baseline}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{text-align:left;padding:9px 14px;border-bottom:1px solid var(--rule-2)}
th{font:500 11px/1 "IBM Plex Mono",monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-2)}
td.n{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;text-align:right}
tr.hi td{background:var(--track-soft)}
.scroll{overflow-x:auto;border:1px solid var(--rule-2);border-radius:3px;background:var(--panel)}
.note{border-left:2px solid var(--track);padding:2px 0 2px 16px;color:var(--ink-2);
  max-width:66ch;font-size:15px}
footer{color:var(--ink-2);font-size:13.5px;border-top:1px solid var(--rule);padding-top:22px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>'''

BODY = r'''<div class="wrap">
<header>
  <p class="eyebrow">Score following on sheet-music images &middot; MSMD-Rec, real room microphone</p>
  <h1>Following the Score</h1>
  <p class="lede">A frozen detector proposes candidate positions on the page; a decoder
  picks one per onset. Press play on any excerpt below and watch where it thinks it is.
  Nothing here is retrained &mdash; the released checkpoint is untouched, and the
  9,697-parameter selector on top of it never saw these recordings.</p>
  <div class="figures">
    <div class="fig"><b class="plain">79.9</b><span>cyolo_sb baseline<br>pct@0.5&nbsp;s, room</span></div>
    <div class="fig"><b>91.4</b><span>ours<br>+11.5, 14 of 16 pieces improved</span></div>
    <div class="fig"><b class="plain">96.0</b><span>ceiling for any causal tracker<br>over these candidates</span></div>
    <div class="fig"><b class="plain">7.3%</b><span>of errors are musically ambiguous<br>the rest are simply wrong</span></div>
  </div>
</header>

<section>
  <p class="eyebrow">Four excerpts</p>
  <h2>What the tracker actually does</h2>
  <p class="lede">Each panel is one page of one performance. The hollow ring is the true
  position, the filled dot is our prediction &mdash; teal inside the half-second
  threshold, red outside. The strip below the score is the error over time. These four
  are not a highlight reel: they are the biggest gain, the best final result, the worst
  piece in the set, and a piece where our learned selector is beaten by the simpler
  hand-written rule.</p>
  <div id="cases" style="display:flex;flex-direction:column;gap:26px"></div>
</section>

<section>
  <p class="eyebrow">Failure analysis &middot; 4,149 onsets, 561 outside threshold</p>
  <h2>The errors are not ambiguity</h2>
  <p class="lede">The obvious hypothesis is that a tracker fails where the music repeats,
  because the image genuinely cannot say which repeat you are in. We tested it directly:
  for every error, compare the notes sounding at the position chosen against the notes at
  the true position.</p>
  <div class="scroll"><table>
    <thead><tr><th>Failure</th><th class="n">Frames</th><th class="n">Share</th><th>What it is</th></tr></thead>
    <tbody>
      <tr><td>Timing drift</td><td class="n">337</td><td class="n">60.1%</td>
        <td>right staff, within two bars, the clock slipped</td></tr>
      <tr><td>Wrong staff line</td><td class="n">213</td><td class="n">38.0%</td>
        <td>correct horizontally, jumped to a neighbouring system</td></tr>
      <tr><td>Gross</td><td class="n">11</td><td class="n">2.0%</td>
        <td>right staff, far away</td></tr>
      <tr class="hi"><td>Musically ambiguous</td><td class="n">41</td><td class="n">7.3%</td>
        <td>identical pitches at both positions &mdash; unfixable from one frame</td></tr>
    </tbody>
  </table></div>
  <p class="note">Median pitch overlap between the chosen and true positions is
  <b>0.000</b>, and 54.9% of errors land somewhere with <i>no notes in common</i> at all.
  So the tracker is not being fooled by a passage that sounds the same. It is simply
  landing in the wrong place, and 99% of the time that place is nearby &mdash; a slipped
  clock or the staff above. That argues against reaching for memory or a long-horizon
  policy to resolve repeats: in this repertoire there is almost nothing to resolve.</p>
</section>

<section>
  <p class="eyebrow">Every piece in the test set &middot; real room microphone</p>
  <h2>Where it wins and where it does not</h2>
  <div class="scroll"><table>
    <thead><tr><th>Piece</th><th class="n">onsets</th><th class="n">cyolo_sb</th>
      <th class="n">+ decoder</th><th class="n">+ selector</th><th class="n">&Delta;</th></tr></thead>
    <tbody>
      <tr class="hi"><td>Chopin, Nocturne Op.&thinsp;9 No.&thinsp;1</td><td class="n">1238</td><td class="n">63.7</td><td class="n">77.3</td><td class="n">87.5</td><td class="n">+23.8</td></tr>
      <tr><td>Schumann, Pauvre Orpheline</td><td class="n">107</td><td class="n">66.4</td><td class="n">76.6</td><td class="n">81.3</td><td class="n">+15.0</td></tr>
      <tr class="hi"><td>Schumann, Melodie Op.&thinsp;68 No.&thinsp;1</td><td class="n">175</td><td class="n">86.3</td><td class="n">91.4</td><td class="n">99.4</td><td class="n">+13.1</td></tr>
      <tr><td>Schumann, Cavalier Sauvage</td><td class="n">154</td><td class="n">88.3</td><td class="n">93.5</td><td class="n">98.7</td><td class="n">+10.4</td></tr>
      <tr><td>Bach, Prelude BWV&thinsp;924a</td><td class="n">212</td><td class="n">81.6</td><td class="n">87.3</td><td class="n">91.5</td><td class="n">+9.9</td></tr>
      <tr><td>Bach, Sinfonia 11 BWV&thinsp;797</td><td class="n">386</td><td class="n">83.2</td><td class="n">87.3</td><td class="n">91.2</td><td class="n">+8.0</td></tr>
      <tr><td>Mozart, KV&thinsp;331 Var.&thinsp;1</td><td class="n">214</td><td class="n">86.9</td><td class="n">86.4</td><td class="n">94.9</td><td class="n">+7.9</td></tr>
      <tr><td>Bach, BWV&thinsp;117a</td><td class="n">157</td><td class="n">86.6</td><td class="n">93.6</td><td class="n">93.6</td><td class="n">+7.0</td></tr>
      <tr><td>Schumann, Sans Titre</td><td class="n">179</td><td class="n">85.5</td><td class="n">87.2</td><td class="n">91.6</td><td class="n">+6.1</td></tr>
      <tr><td>Bach, Anna Magdalena 3</td><td class="n">195</td><td class="n">93.3</td><td class="n">93.8</td><td class="n">99.0</td><td class="n">+5.6</td></tr>
      <tr><td>Bach, Anna Magdalena 7</td><td class="n">221</td><td class="n">94.1</td><td class="n">98.2</td><td class="n">98.2</td><td class="n">+4.1</td></tr>
      <tr><td>Bach, BWV&thinsp;120</td><td class="n">113</td><td class="n">94.7</td><td class="n">97.3</td><td class="n">98.2</td><td class="n">+3.5</td></tr>
      <tr><td>Bach, Partita BWV&thinsp;830</td><td class="n">472</td><td class="n">91.3</td><td class="n">93.9</td><td class="n">94.1</td><td class="n">+2.8</td></tr>
      <tr><td>Bach, French Suite 6 Menuet</td><td class="n">140</td><td class="n">97.9</td><td class="n">97.1</td><td class="n">100.0</td><td class="n">+2.1</td></tr>
      <tr class="hi"><td>Schumann, Premier Chagrin</td><td class="n">130</td><td class="n">85.4</td><td class="n">92.3</td><td class="n">84.6</td><td class="n">&minus;0.8</td></tr>
      <tr class="hi"><td>Mussorgsky, Promenade 3</td><td class="n">56</td><td class="n">46.4</td><td class="n">46.4</td><td class="n">39.3</td><td class="n">&minus;7.1</td></tr>
      <tr><td><b>All onsets (micro)</b></td><td class="n"><b>4149</b></td><td class="n"><b>79.9</b></td><td class="n"><b>86.5</b></td><td class="n"><b>91.4</b></td><td class="n"><b>+11.5</b></td></tr>
    </tbody>
  </table></div>
  <p class="note">Fourteen of sixteen pieces improve. The two that do not are worth more
  than the fourteen that do: on <b>Premier Chagrin</b> the learned selector is beaten by
  the hand-written rule it was meant to replace, and on <b>Promenade 3</b> &mdash; 56
  onsets, the shortest and hardest piece &mdash; it makes a bad result worse. Both are in
  the panels above.</p>
</section>

<section>
  <p class="eyebrow">Same configuration, three recordings of the same performances</p>
  <h2>It holds across recording conditions</h2>
  <div class="scroll"><table>
    <thead><tr><th>Recording</th><th class="n">cyolo_sb</th><th class="n">+ decoder</th>
      <th class="n">+ selector</th></tr></thead>
    <tbody>
      <tr class="hi"><td>Room microphone</td><td class="n">79.9</td><td class="n">86.5</td><td class="n">91.4</td></tr>
      <tr><td>Direct pickup</td><td class="n">83.6</td><td class="n">89.1</td><td class="n">93.3</td></tr>
      <tr><td>Synthetic, same pieces</td><td class="n">87.2</td><td class="n">90.6</td><td class="n">93.4</td></tr>
      <tr><td>Synthetic, full 94-piece set</td><td class="n">89.3</td><td class="n">91.6</td><td class="n">&mdash;</td></tr>
    </tbody>
  </table></div>
  <p class="note">One frozen checkpoint throughout. The decoder adds no parameters; the
  selector adds 9,697, fitted on the training split and selected on held-out validation
  &mdash; none of these recordings informed either choice. Against the baseline the gain
  is +11.5 on room, resolvable under a bootstrap clustered by piece
  (p&nbsp;&lt;&nbsp;0.001). For reference, the published variant that reaches 86.5 on room
  needs augmentation data that was never released.</p>
</section>

<footer>Positions, ground truth and thresholds are the evaluation harness&rsquo;s own,
recorded frame by frame during the scored run. Audio is the original room-microphone
recording, cut to the span the tracker spent on the page shown.</footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const css = getComputedStyle(document.documentElement);
const C = () => ({
  track: css.getPropertyValue('--track').trim(),
  miss: css.getPropertyValue('--miss').trim(),
  truth: css.getPropertyValue('--truth').trim(),
  rule: css.getPropertyValue('--rule').trim(),
  ink: css.getPropertyValue('--ink-2').trim(),
});
const fmt = s => (s<0?0:s).toFixed(1).padStart(4,'0') + 's';
function lower(arr, x){let lo=0,hi=arr.length-1,r=-1;
  while(lo<=hi){const m=(lo+hi)>>1; if(arr[m]<=x){r=m;lo=m+1;}else hi=m-1;} return r;}

DATA.cases.forEach(c => { try {
  const el = document.createElement('article');
  el.className = 'case';
  el.innerHTML = `
    <div class="case-head">
      <h2>${c.title}</h2>
      <p class="why">${c.why} &middot; page ${c.page + 1}</p>
      <p class="clock">page: ${c.page_base.toFixed(1)}% &rarr; <b style="color:var(--track)">${c.page_ours.toFixed(1)}%</b></p>
    </div>
    <div class="score"><img src="${c.img}" alt="Score page ${c.page + 1} of ${c.title}" width="${c.w}" height="${c.h}"><canvas width="${c.w}" height="${c.h}"></canvas></div>
    <div class="transport">
      <div class="row">
        <button type="button" data-play>Play</button>
        <button type="button" class="ghost" data-base aria-pressed="false">Show baseline</button>
        <span class="clock" data-clock>0.0s</span>
        <span class="legend">
          <span><i style="background:transparent;border:2px solid var(--truth)"></i>true position</span>
          <span><i style="background:var(--track)"></i>ours, within 0.5&nbsp;s</span>
          <span><i style="background:var(--miss)"></i>ours, outside</span>
        </span>
      </div>
      <div class="strip"><canvas></canvas></div>
      <audio src="${c.audio}" preload="metadata"></audio>
    </div>`;
  document.getElementById('cases').appendChild(el);

  const img = el.querySelector('img'), cv = el.querySelector('.score canvas');
  const strip = el.querySelector('.strip canvas'), audio = el.querySelector('audio');
  const btn = el.querySelector('[data-play]'), bbtn = el.querySelector('[data-base]');
  const clock = el.querySelector('[data-clock]');
  const g = cv.getContext('2d'); let showBase = false;

  const off = document.createElement('canvas');
  // The strip is static except when the baseline toggles or the panel resizes,
  // so it is rendered once into an offscreen canvas and blitted each frame.
  // Rebuilding it inside the animation loop meant resizing four canvases sixty
  // times a second, which is what locked the page up.
  function drawStrip(){
    const r = strip.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * devicePixelRatio));
    const h = Math.max(1, Math.round(r.height * devicePixelRatio));
    strip.width = off.width = w; strip.height = off.height = h;
    const s = off.getContext('2d'), W = w, H = h, col = C();
    s.clearRect(0,0,W,H);
    const t0 = c.t[0], t1 = c.t[c.t.length-1], span = Math.max(t1-t0, 1e-6);
    const cap = 3, y = e => H - Math.min(e, cap)/cap * (H-8) - 4;
    s.strokeStyle = col.rule; s.lineWidth = devicePixelRatio;
    s.beginPath(); s.moveTo(0, y(0.5)); s.lineTo(W, y(0.5)); s.stroke();
    s.setLineDash([]);
    const line = (ts, es, colr, w) => {
      s.strokeStyle = colr; s.lineWidth = w*devicePixelRatio; s.beginPath();
      ts.forEach((tt,i)=>{const x=(tt-t0)/span*W; i?s.lineTo(x,y(es[i])):s.moveTo(x,y(es[i]));});
      s.stroke();
    };
    if (showBase) line(c.bt, c.be, col.truth, 1);
    line(c.t, c.e, col.track, 1.4);
    s.fillStyle = col.ink; s.font = `${11*devicePixelRatio}px "IBM Plex Mono",monospace`;
    s.fillText('0.5 s', 4*devicePixelRatio, y(0.5) - 4*devicePixelRatio);
    return {t0, span};
  }
  let stripGeom = null;

  function draw(){
    const t = audio.currentTime + c.t0, col = C();
    g.clearRect(0,0,cv.width,cv.height);
    const i = lower(c.t, t);
    // trail of where it has been
    if (i >= 0){
      g.lineWidth = 2; g.strokeStyle = col.track + '55'; g.beginPath();
      for (let k=0;k<=i;k++) k?g.lineTo(c.px[k],c.py[k]):g.moveTo(c.px[k],c.py[k]);
      g.stroke();
    }
    const dot=(x,y,r,fill,stroke)=>{g.beginPath();g.arc(x,y,r,0,6.2832);
      if(fill){g.fillStyle=fill;g.fill();} if(stroke){g.lineWidth=3;g.strokeStyle=stroke;g.stroke();}};
    if (i >= 0){
      dot(c.gx[i], c.gy[i], 11, null, col.truth);
      if (showBase){
        const j = lower(c.bt, t);
        if (j >= 0) dot(c.bx[j], c.by[j], 6, col.truth + '99', null);
      }
      dot(c.px[i], c.py[i], 8, c.e[i] <= 0.5 ? col.track : col.miss, null);
    }
    if (stripGeom){
      const s = strip.getContext('2d');
      s.clearRect(0, 0, strip.width, strip.height);
      s.drawImage(off, 0, 0);
      const x = (t - stripGeom.t0) / stripGeom.span * strip.width;
      s.strokeStyle = col.ink; s.lineWidth = devicePixelRatio;
      s.beginPath(); s.moveTo(x,0); s.lineTo(x,strip.height); s.stroke();
    }
    clock.textContent = fmt(t - c.t[0]) + (i>=0 ? `   err ${c.e[i].toFixed(2)}s` : '');
    if (!audio.paused) requestAnimationFrame(draw);
  }
  const safeDraw = () => { try { draw(); } catch (e) { console.error('draw', e); } };

  btn.addEventListener('click', () => {
    document.querySelectorAll('audio').forEach(a => { if (a !== audio) a.pause(); });
    if (audio.paused){ audio.play(); btn.textContent = 'Pause'; requestAnimationFrame(safeDraw); }
    else { audio.pause(); btn.textContent = 'Play'; }
  });
  audio.addEventListener('pause', () => btn.textContent = 'Play');
  audio.addEventListener('play',  () => btn.textContent = 'Pause');
  audio.addEventListener('ended', () => { btn.textContent = 'Play'; });
  bbtn.addEventListener('click', () => {
    showBase = !showBase; bbtn.setAttribute('aria-pressed', showBase);
    bbtn.textContent = showBase ? 'Hide baseline' : 'Show baseline';
    stripGeom = drawStrip(); safeDraw();
  });
  el.querySelector('.strip').addEventListener('click', ev => {
    const r = ev.currentTarget.getBoundingClientRect();
    const f = (ev.clientX - r.left) / r.width;
    audio.currentTime = Math.max(0, Math.min(audio.duration || 0,
      f * (c.t[c.t.length-1] - c.t[0]) + (c.t[0] - c.t0)));
    safeDraw();
  });
  const boot = () => { try { stripGeom = drawStrip(); safeDraw(); } catch (e) { console.error('boot', e); } };
  if (img.complete) boot(); else img.addEventListener('load', boot);
  } catch (e) { console.error('case', c && c.id, e); }
  let rt = 0;
  addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { stripGeom = drawStrip(); safeDraw(); }, 120); });
});
</script>'''

html = HEAD + '\n' + BODY.replace('__PAYLOAD__', pay)
p = os.path.join(OUT, 'score_demo.html')
open(p, 'w').write(html)
print(f'wrote {p}  ({os.path.getsize(p)/1e6:.2f} MB)')
