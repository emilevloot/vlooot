"""
Make tune_report.html from tune_history.json: graphs of the tuning run.

    python tune_report.py

tune.py runs this by itself at the end. Open tune_report.html in a browser.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(HERE, "tune_history.json")
REPORT_FILE = os.path.join(HERE, "tune_report.html")


def make():
    hist = json.load(open(HISTORY_FILE, encoding="utf-8"))
    data = json.dumps(hist).replace("</", "<\\/")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("/*DATA*/null", data))
    return REPORT_FILE


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bot Tuning Report</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7;
    --surface-1: #fcfcfb;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --series-2: #eb6834;
    --good: #006300;
    --bad: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d;
      --surface-1: #1a1a19;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-2: #d95926;
      --good: #0ca30c;
      --bad: #e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --border: rgba(255,255,255,0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --good: #0ca30c;
    --bad: #e66767;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 16px 48px; background: var(--page); color: var(--text-primary);
    font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 1180px; margin: 0 auto; display: grid; gap: 20px; }
  h1 { font-size: 26px; margin: 0; font-weight: 650; }
  h2 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  p.sub { margin: 0 0 12px; color: var(--text-secondary); max-width: 80ch; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px; min-width: 0; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
  .tile .label { color: var(--text-secondary); font-size: 13px; }
  .tile .value { font-size: 30px; font-weight: 600; margin-top: 2px; }
  .tile .delta { font-size: 13px; color: var(--text-secondary); }
  .up { color: var(--good); } .down { color: var(--bad); }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .key-line { width: 18px; height: 2px; background: var(--series-1); border-radius: 1px; }
  .key-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--series-2); }
  .key-ring { width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--text-primary); }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  svg text { fill: var(--muted); font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
  svg .lbl { fill: var(--text-secondary); font-size: 12px; }
  .grid line { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .champ { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .champ-dot { fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; }
  .var-dot { fill: var(--series-2); stroke: var(--surface-1); stroke-width: 2; }
  .new-ring { fill: none; stroke: var(--text-primary); stroke-width: 2; }
  .cross { stroke: var(--axis); stroke-width: 1; }
  .ref { stroke: var(--muted); stroke-width: 1; }
  .multiples { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; }
  .panel { border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px 6px; min-width: 0; }
  .panel .name { font-size: 13px; font-weight: 600; font-family: ui-monospace, Consolas, monospace; overflow-wrap: anywhere; }
  .panel .vals { font-size: 12px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
  .panel.same { opacity: .6; }
  .tip {
    position: fixed; pointer-events: none; z-index: 10; background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 9px; font-size: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,.18); font-variant-numeric: tabular-nums; max-width: 260px;
  }
  .tip b { font-weight: 600; }
  details summary { cursor: pointer; font-weight: 600; }
  .tablewrap { overflow-x: auto; margin-top: 10px; }
  table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
  th, td { padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--text-secondary); font-weight: 600; }
  .empty { color: var(--text-secondary); }
</style>
</head>
<body>
<main>
  <header>
    <h1>Bot tuning report</h1>
    <p class="sub" id="intro"></p>
  </header>
  <section class="tiles" id="tiles"></section>
  <section class="card">
    <h2>Average score per game</h2>
    <p class="sub">Each round, a few variations of the best weights (the champion) play 3-player games against two champions on the same boards. The line is the champion's average score in that round; the dots are the variations. A ring marks a variation that was confirmed and became the new champion.</p>
    <div class="legend">
      <span><i class="key-line"></i>Champion</span>
      <span><i class="key-dot"></i>Variation</span>
      <span><i class="key-ring"></i>New champion</span>
    </div>
    <svg id="scoreChart" role="img" aria-label="Average score per game by round"></svg>
  </section>
  <section class="card">
    <h2>Points ahead of the champion</h2>
    <p class="sub">How many points per game each variation scored more than the champions it played against. Above 0 is better; only variations clearly above 0 in a second, bigger test are kept.</p>
    <svg id="advChart" role="img" aria-label="Points ahead of the champion by round"></svg>
  </section>
  <section class="card">
    <h2>The weights over the rounds</h2>
    <p class="sub">The champion's value of every weight after each round. The height of each panel is the range the tuner may use; the thin gray line is the starting value. Weights that changed come first.</p>
    <div class="multiples" id="multiples"></div>
  </section>
  <section class="card">
    <details>
      <summary>All rounds as a table</summary>
      <div class="tablewrap" id="table"></div>
    </details>
  </section>
</main>
<div class="tip" id="tip" hidden></div>
<script>
const H = /*DATA*/null;
const R = H.rounds;
const $ = id => document.getElementById(id);
const fmt = (v, d = 1) => (v == null || isNaN(v)) ? '–' : Number(v).toFixed(d);
const sgn = (v, d = 1) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const NS = 'http://www.w3.org/2000/svg';

// ---------- tooltip ----------
const tip = $('tip');
function showTip(e, html) {
  tip.innerHTML = html; tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > innerHeight - 8) y = e.clientY - h - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => { tip.hidden = true; };

// ---------- helpers ----------
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || mag * 10;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return { ticks: out, lo: Math.floor(lo / step) * step, hi: Math.ceil(hi / step) * step };
}

// ---------- intro + tiles ----------
const accepted = R.filter(r => r.accepted);
const mins = R.length ? R[R.length - 1].time_s / 60 : 0;
$('intro').textContent = R.length
  ? `${R.length} rounds in ${fmt(mins, 0)} minutes, ${R.reduce((s, r) => s + r.variations.length, 0)} variations tried, ${accepted.length} new champion${accepted.length === 1 ? '' : 's'}.`
  : 'No rounds yet.';
const F = H.final;
const tiles = [];
if (F) {
  tiles.push(['Tuned vs original: win rate', (100 * F.win).toFixed(1) + '%', `fair share is 33.3% (${F.games} games)`, F.win > 1 / 3 ? 'up' : 'down']);
  tiles.push(['Points per game vs original', sgn(F.adv), `± ${fmt(F.ci)} (95% range)`, F.adv - F.ci > 0 ? 'up' : F.adv + F.ci < 0 ? 'down' : '']);
  tiles.push(['Average score: tuned', fmt(F.score), `original: ${fmt(F.original_score)}`, '']);
}
tiles.push(['New champions', String(accepted.length), `out of ${R.length} rounds`, '']);
$('tiles').innerHTML = tiles.map(([l, v, d, c]) =>
  `<div class="card tile"><div class="label">${l}</div><div class="value">${v}</div><div class="delta ${c}">${d}</div></div>`).join('');

// ---------- round chart (shared by score and advantage charts) ----------
function roundChart(svg, opts) {
  const W = 1100, Hh = 300, m = { l: 48, r: 16, t: 12, b: 34 };
  svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`);
  if (!R.length) { svg.innerHTML = '<text x="10" y="20">No rounds yet.</text>'; return; }
  const vals = [];
  R.forEach(r => opts.points(r).forEach(p => vals.push(p.y)));
  if (opts.line) R.forEach(r => vals.push(opts.line(r)));
  if (opts.zero) vals.push(0);
  const pad = (Math.max(...vals) - Math.min(...vals)) * 0.08 || 1;
  const yt = niceTicks(Math.min(...vals) - pad, Math.max(...vals) + pad);
  const x0 = R[0].round, x1 = R[R.length - 1].round;
  const X = v => m.l + (x1 === x0 ? (W - m.l - m.r) / 2 : (v - x0) / (x1 - x0) * (W - m.l - m.r));
  const Y = v => m.t + (1 - (v - yt.lo) / (yt.hi - yt.lo)) * (Hh - m.t - m.b);
  let s = '<g class="grid">';
  yt.ticks.forEach(t => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(t)}" y2="${Y(t)}"/>`; });
  s += '</g>';
  yt.ticks.forEach(t => { s += `<text x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${t}</text>`; });
  if (opts.zero) s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${Y(0)}" y2="${Y(0)}" style="stroke:var(--muted)"/>`;
  s += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${Hh - m.b}" y2="${Hh - m.b}"/>`;
  const every = Math.max(1, Math.ceil(R.length / 12));
  R.forEach((r, i) => { if (i % every === 0 || i === R.length - 1) s += `<text x="${X(r.round)}" y="${Hh - m.b + 18}" text-anchor="middle">${r.round}</text>`; });
  s += `<text class="lbl" x="${(m.l + W - m.r) / 2}" y="${Hh - 2}" text-anchor="middle">round</text>`;
  s += `<line class="cross" id="${svg.id}X" y1="${m.t}" y2="${Hh - m.b}" visibility="hidden"/>`;
  if (opts.line) {
    s += `<path class="champ" d="${R.map((r, i) => (i ? 'L' : 'M') + X(r.round) + ' ' + Y(opts.line(r))).join('')}"/>`;
    R.forEach(r => { s += `<circle class="champ-dot" cx="${X(r.round)}" cy="${Y(opts.line(r))}" r="4"/>`; });
    const last = R[R.length - 1];
    s += `<text class="lbl" x="${X(last.round) - 6}" y="${Y(opts.line(last)) - 10}" text-anchor="end">${fmt(opts.line(last))}</text>`;
  }
  R.forEach(r => opts.points(r).forEach(p => {
    const jitter = (p.i - (opts.points(r).length - 1) / 2) * 2.2;
    s += `<circle class="var-dot" cx="${X(r.round) + jitter}" cy="${Y(p.y)}" r="4"/>`;
    if (p.ring) s += `<circle class="new-ring" cx="${X(r.round) + jitter}" cy="${Y(p.y)}" r="8"/>`;
  }));
  svg.innerHTML = s;
  // hover: nearest round -> crosshair + tooltip
  const cross = svg.querySelector('.cross');
  svg.onmousemove = e => {
    const pt = svg.getBoundingClientRect();
    const vx = (e.clientX - pt.left) / pt.width * W;
    let best = R[0];
    R.forEach(r => { if (Math.abs(X(r.round) - vx) < Math.abs(X(best.round) - vx)) best = r; });
    cross.setAttribute('x1', X(best.round)); cross.setAttribute('x2', X(best.round));
    cross.setAttribute('visibility', 'visible');
    showTip(e, opts.tip(best));
  };
  svg.onmouseleave = () => { cross.setAttribute('visibility', 'hidden'); hideTip(); };
}

function bestIndex(r) { let b = 0; r.variations.forEach((v, i) => { if (v.adv > r.variations[b].adv) b = i; }); return b; }

roundChart($('scoreChart'), {
  line: r => r.champ_score,
  points: r => { const b = bestIndex(r); return r.variations.map((v, i) => ({ y: v.score, i, ring: r.accepted && i === b })); },
  tip: r => `<b>Round ${r.round}</b><br>Champion: ${fmt(r.champ_score)} pts per game<br>Variations: ${r.variations.map(v => fmt(v.score)).join(', ')}` +
    (r.check ? `<br>Check of the best: ${fmt(r.check.score)} vs ${fmt(r.check.champ_score)}` : '') +
    (r.accepted ? '<br><b>New champion</b>' : ''),
});

roundChart($('advChart'), {
  zero: true,
  points: r => { const b = bestIndex(r); return r.variations.map((v, i) => ({ y: v.adv, i, ring: r.accepted && i === b })); },
  tip: r => `<b>Round ${r.round}</b><br>Variations: ${r.variations.map(v => sgn(v.adv)).join(', ')} pts` +
    (r.check ? `<br>Check: ${sgn(r.check.adv)} ± ${fmt(r.check.ci)} pts, wins ${(100 * r.check.win).toFixed(0)}%` : '<br>No check (none ahead)') +
    (r.accepted ? '<br><b>New champion</b>' : ''),
});

// ---------- small multiples: one panel per weight ----------
const names = Object.keys(H.start_weights);
const last = R.length ? R[R.length - 1].weights : H.start_weights;
const changed = n => last[n] !== H.start_weights[n];
names.sort((a, b) => (changed(b) - changed(a)) || a.localeCompare(b));
$('multiples').innerHTML = names.map(n => {
  const [lo, hi, whole] = H.ranges[n];
  const d = whole ? 0 : 2;
  const series = [H.start_weights[n]].concat(R.map(r => r.weights[n]));
  const W = 200, Hh = 64, pl = 2, pr = 2, pt = 4, pb = 4;
  const X = i => pl + (series.length === 1 ? 0 : i / (series.length - 1) * (W - pl - pr));
  const Y = v => pt + (1 - (v - lo) / (hi - lo || 1)) * (Hh - pt - pb);
  // step line: the value holds until the round it changes
  let dd = `M${X(0)} ${Y(series[0])}`;
  for (let i = 1; i < series.length; i++) dd += `H${X(i)}V${Y(series[i])}`;
  const svg = `<svg viewBox="0 0 ${W} ${Hh}" preserveAspectRatio="none" data-w="${n}" style="height:64px">
    <line class="ref" x1="${pl}" x2="${W - pr}" y1="${Y(series[0])}" y2="${Y(series[0])}" vector-effect="non-scaling-stroke"/>
    <line class="axis" x1="${pl}" x2="${W - pr}" y1="${Hh - pb}" y2="${Hh - pb}" vector-effect="non-scaling-stroke"/>
    <path class="champ" d="${dd}" vector-effect="non-scaling-stroke"/></svg>`;
  return `<div class="panel${changed(n) ? '' : ' same'}">
    <div class="name">${n}</div>
    <div class="vals">${fmt(H.start_weights[n], d)} → <b>${fmt(last[n], d)}</b> <span style="color:var(--muted)">(range ${lo}–${hi})</span></div>
    ${svg}</div>`;
}).join('');
document.querySelectorAll('.multiples svg').forEach(svg => {
  const n = svg.dataset.w;
  const series = [H.start_weights[n]].concat(R.map(r => r.weights[n]));
  const d = H.ranges[n][2] ? 0 : 3;
  svg.onmousemove = e => {
    const b = svg.getBoundingClientRect();
    const i = Math.round((e.clientX - b.left) / b.width * (series.length - 1));
    const k = Math.max(0, Math.min(series.length - 1, i));
    showTip(e, `<b>${n}</b><br>${k === 0 ? 'Start' : 'After round ' + R[k - 1].round}: ${fmt(series[k], d)}`);
  };
  svg.onmouseleave = hideTip;
});

// ---------- table view ----------
$('table').innerHTML = R.length ? `<table>
  <tr><th>Round</th><th>Champion score</th><th>Best variation score</th><th>Best ahead by</th><th>Check</th><th>New champion</th></tr>
  ${R.map(r => { const b = r.variations[bestIndex(r)]; return `<tr><td>${r.round}</td><td>${fmt(r.champ_score)}</td><td>${fmt(b.score)}</td><td>${sgn(b.adv)}</td>
    <td>${r.check ? sgn(r.check.adv) + ' ± ' + fmt(r.check.ci) : '–'}</td><td>${r.accepted ? 'yes' : ''}</td></tr>`; }).join('')}
</table>` : '<p class="empty">No rounds yet.</p>';
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Wrote", make())
