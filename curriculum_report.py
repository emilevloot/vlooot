"""
Make curriculum_report.html: graphs of the curriculum training runs.

    python curriculum_report.py            # compares all runs it finds

curriculum.py runs this by itself at the end. Open the file in a browser.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "curriculum_report.html")
STAGES = ["resources", "buildings", "sites", "longships", "full"]
LABELS = {"cur": "Run 1: basic features, final results only",
          "c2": "Run 2: helper features + TD learning",
          "c3": "Run 3: land encoding v3 (placement gains, shopping list), GPU self-play"}


def _old_format(res):
    """Run 1 stored its results differently; turn them into steps."""
    out = {"prefix": "cur", "stages": {}, "final": {}}
    for st, r in res.get("stages", {}).items():
        steps = []
        if "arena_before_selfplay" in r:
            steps.append({"model": "cur_%s_a" % st, "kind": "greedy data",
                          "arena": r["arena_before_selfplay"]})
        if "arena_after_selfplay" in r:
            steps.append({"model": "cur_%s" % st, "kind": "self-play round 1",
                          "arena": r["arena_after_selfplay"]})
        for k, it in enumerate(r.get("iterations", []), start=2):
            steps.append({"model": it["model"], "kind": "self-play round %d" % k,
                          "arena": it["arena"]})
        out["stages"][st] = {"steps": steps}
    if "v1_full_game" in res:
        out["final"]["v1 (no curriculum) vs greedy"] = res["v1_full_game"]
    return out


def load_runs():
    runs = []
    for f in sorted(glob.glob(os.path.join(HERE, "curriculum_*_results.json"))):
        prefix = os.path.basename(f)[len("curriculum_"):-len("_results.json")]
        res = json.load(open(f))
        if "steps" not in json.dumps(res):
            res = _old_format(res)
        res["prefix"] = prefix
        res["label"] = LABELS.get(prefix, "Run " + prefix)
        runs.append(res)
    runs.sort(key=lambda r: (r["prefix"] != "cur", r["prefix"]))
    return runs


def make():
    data = json.dumps({"runs": load_runs(), "stages": STAGES}).replace("</", "<\\/")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("/*DATA*/null", data))
    return REPORT


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Curriculum Training Report</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6; --series-2: #eb6834; --good: #006300; --bad: #d03b3b;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --series-1: #3987e5; --series-2: #d95926; --good: #0ca30c; --bad: #e66767;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --good: #0ca30c; --bad: #e66767;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px 16px 48px; background: var(--page); color: var(--text-primary);
    font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1180px; margin: 0 auto; display: grid; gap: 20px; }
  h1 { font-size: 26px; margin: 0; font-weight: 650; }
  h2 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  h3 { font-size: 14px; margin: 0 0 2px; font-weight: 600; }
  p.sub { margin: 0 0 12px; color: var(--text-secondary); max-width: 85ch; }
  .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px; min-width: 0; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
  .tile .label { color: var(--text-secondary); font-size: 13px; }
  .tile .value { font-size: 30px; font-weight: 600; margin-top: 2px; }
  .tile .delta { font-size: 13px; color: var(--text-secondary); }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin-bottom: 8px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .key { width: 18px; height: 2px; border-radius: 1px; }
  .grid5 { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 12px; }
  .panel { border: 1px solid var(--border); border-radius: 10px; padding: 8px 10px; min-width: 0; }
  .panel .note { font-size: 12px; color: var(--text-secondary); }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  svg text { fill: var(--muted); font: 10px system-ui, -apple-system, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
  .gridl { stroke: var(--grid); stroke-width: 1; }
  .ref { stroke: var(--muted); stroke-width: 1; }
  .ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .dot { stroke: var(--surface-1); stroke-width: 2; }
  .tip { position: fixed; pointer-events: none; z-index: 10; background: var(--surface-1); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 8px; padding: 6px 9px; font-size: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,.18); font-variant-numeric: tabular-nums; max-width: 280px; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
  th, td { padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  th { color: var(--text-secondary); font-weight: 600; }
  details summary { cursor: pointer; font-weight: 600; }
</style>
</head>
<body>
<main>
  <header>
    <h1>Curriculum training report</h1>
    <p class="sub">The value network learns Looot in 5 steps, each adding rules: resources, buildings, construction sites, longships, the full game. After training on greedy games it plays itself for a few rounds. Every point below is a network playing 2-player games against the tuned greedy player under that step's rules.</p>
  </header>
  <section class="tiles" id="tiles"></section>
  <section class="card">
    <h2>Win rate against greedy, per step</h2>
    <p class="sub">The first point of each panel is the network trained on greedy games, the next ones are self-play rounds. 50% (the gray line) means as strong as greedy.</p>
    <div class="legend" id="legend"></div>
    <div class="grid5" id="winPanels"></div>
  </section>
  <section class="card">
    <h2>Points lost to unfilled longships</h2>
    <p class="sub">Average penalty per game (−5 per unfilled longship) in the steps that have longships. The greedy player loses about 1–2 points here.</p>
    <div class="grid5" id="penPanels"></div>
  </section>
  <section class="card">
    <h2>Final tests (full game)</h2>
    <div class="tablewrap" id="finals"></div>
  </section>
  <section class="card">
    <details>
      <summary>Every step as a table</summary>
      <div class="tablewrap" id="table" style="margin-top:10px"></div>
    </details>
  </section>
</main>
<div class="tip" id="tip" hidden></div>
<script>
const D = /*DATA*/null;
const $ = id => document.getElementById(id);
const COL = ['var(--series-2)', 'var(--series-1)'];   // run 1 orange, run 2 blue
const colOf = i => COL[Math.max(0, COL.length - D.runs.length + i)];
const pct = v => (100 * v).toFixed(1) + '%';
const sgn = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1);
const tip = $('tip');
function showTip(e, html) {
  tip.innerHTML = html; tip.hidden = false;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + tip.offsetWidth > innerWidth - 8) x = e.clientX - tip.offsetWidth - 14;
  if (y + tip.offsetHeight > innerHeight - 8) y = e.clientY - tip.offsetHeight - 14;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => { tip.hidden = true; };

// legend (always, since there can be 2 runs)
$('legend').innerHTML = D.runs.map((r, i) => `<span><i class="key" style="background:${colOf(i)}"></i>${r.label}</span>`).join('');

// ---------- tiles ----------
const last = D.runs[D.runs.length - 1];
const tiles = [];
for (const [label, v] of Object.entries(last.final || {})) {
  tiles.push(`<div class="card tile"><div class="label">${label}</div><div class="value">${pct(v.win)}</div>
    <div class="delta">± ${(100 * (v.win_ci || 0)).toFixed(1)} · margin ${sgn(v.margin)} · ${v.games} games</div></div>`);
}
for (const st of ['sites', 'longships']) {
  const s = (last.stages[st] || {}).steps || [];
  if (s.length) {
    const b = s.reduce((a, x) => x.arena.win > a.arena.win ? x : a);
    tiles.push(`<div class="card tile"><div class="label">Best in step "${st}"</div><div class="value">${pct(b.arena.win)}</div><div class="delta">${b.model}</div></div>`);
  }
}
$('tiles').innerHTML = tiles.join('');

// ---------- small multiples ----------
function panel(st, key, opts) {
  const series = D.runs.map(r => ((r.stages[st] || {}).steps || []).map((s, i) => ({ i, v: opts.get(s), s })));
  const all = series.flat();
  if (!all.length) return '';
  const W = 220, H = 130, ml = 30, mr = 6, mt = 8, mb = 18;
  const n = Math.max(...series.map(s => s.length));
  let lo = Math.min(...all.map(p => p.v), opts.ref ?? Infinity), hi = Math.max(...all.map(p => p.v), opts.ref ?? -Infinity);
  if (opts.fixed) [lo, hi] = opts.fixed;
  const pad = (hi - lo) * 0.1 || 1; lo -= opts.fixed ? 0 : pad; hi += opts.fixed ? 0 : pad;
  const X = i => ml + (n <= 1 ? (W - ml - mr) / 2 : i / (n - 1) * (W - ml - mr));
  const Y = v => mt + (1 - (v - lo) / (hi - lo)) * (H - mt - mb);
  let s = '';
  opts.ticks(lo, hi).forEach(t => { s += `<line class="gridl" x1="${ml}" x2="${W - mr}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${ml - 4}" y="${Y(t) + 3}" text-anchor="end">${opts.fmt(t)}</text>`; });
  if (opts.ref !== undefined) s += `<line class="ref" x1="${ml}" x2="${W - mr}" y1="${Y(opts.ref)}" y2="${Y(opts.ref)}"/>`;
  for (let i = 0; i < n; i++) s += `<text x="${X(i)}" y="${H - 4}" text-anchor="middle">${i === 0 ? 'greedy' : 'r' + i}</text>`;
  series.forEach((pts, ri) => {
    if (!pts.length) return;
    s += `<path class="ln" style="stroke:${colOf(ri)}" d="${pts.map((p, k) => (k ? 'L' : 'M') + X(p.i) + ' ' + Y(p.v)).join('')}"/>`;
    pts.forEach(p => { s += `<circle class="dot" style="fill:${colOf(ri)}" cx="${X(p.i)}" cy="${Y(p.v)}" r="4" data-r="${ri}" data-i="${p.i}"/>`; });
  });
  return `<div class="panel" data-st="${st}" data-key="${key}"><h3>${st}</h3><div class="note">${opts.note || ''}</div>
    <svg viewBox="0 0 ${W} ${H}">${s}</svg></div>`;
}
const winOpts = { get: s => s.arena.win, ref: 0.5, fixed: [0, 1], fmt: t => Math.round(t * 100) + '%',
  ticks: () => [0, 0.25, 0.5, 0.75, 1] };
$('winPanels').innerHTML = D.stages.map(st => panel(st, 'win', winOpts)).join('');
const penOpts = { get: s => s.arena.penalty, fmt: t => t.toFixed(0),
  ticks: (lo, hi) => { const out = []; const step = hi - lo > 20 ? 10 : 5; for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v); return out; } };
$('penPanels').innerHTML = ['longships', 'full'].map(st => panel(st, 'pen', penOpts)).join('');

// hover on dots
document.querySelectorAll('.panel').forEach(p => {
  p.querySelectorAll('circle').forEach(c => {
    c.addEventListener('mousemove', e => {
      const run = D.runs[+c.dataset.r], st = p.dataset.st;
      const step = run.stages[st].steps[+c.dataset.i], a = step.arena;
      showTip(e, `<b>${run.label}</b><br>${st}: ${step.kind}<br>${step.model}<br>wins ${pct(a.win)}, margin ${sgn(a.margin)}<br>score ${a.score.toFixed(1)}, longship penalty ${a.penalty.toFixed(1)}` +
        (step.test_mae ? `<br>test error ${step.test_mae.toFixed(2)} points` : ''));
    });
    c.addEventListener('mouseleave', hideTip);
    c.style.cursor = 'default';
  });
});

// ---------- finals ----------
const fin = D.runs.flatMap(r => Object.entries(r.final || {}).map(([k, v]) => [r.label, k, v]));
$('finals').innerHTML = fin.length ? `<table><tr><th>Run</th><th>Test</th><th>Wins</th><th>±</th><th>Margin</th><th>Score</th><th>Longship penalty</th><th>ms/turn</th><th>Games</th></tr>
  ${fin.map(([run, k, v]) => `<tr><td>${run}</td><td>${k}</td><td>${pct(v.win)}</td><td>${v.win_ci ? (100 * v.win_ci).toFixed(1) : '–'}</td><td>${sgn(v.margin)}</td><td>${v.score !== undefined ? v.score.toFixed(1) : '–'}</td><td>${v.penalty.toFixed(1)}</td><td>${v.ms_per_turn ? v.ms_per_turn.toFixed(0) : '–'}</td><td>${v.games}</td></tr>`).join('')}</table>`
  : '<p class="sub">No final tests yet.</p>';

// ---------- table ----------
$('table').innerHTML = `<table><tr><th>Run</th><th>Step</th><th>Model</th><th>Wins</th><th>Margin</th><th>Score</th><th>Longship penalty</th><th>Test error</th></tr>
  ${D.runs.flatMap(r => D.stages.flatMap(st => ((r.stages[st] || {}).steps || []).map(s =>
    `<tr><td>${r.prefix}</td><td>${st}: ${s.kind}</td><td>${s.model}</td><td>${pct(s.arena.win)}</td><td>${sgn(s.arena.margin)}</td><td>${s.arena.score.toFixed(1)}</td><td>${s.arena.penalty.toFixed(1)}</td><td>${s.test_mae ? s.test_mae.toFixed(2) : '–'}</td></tr>`))).join('')}</table>`;
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Wrote", make())
