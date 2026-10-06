// The neural network in the background (a web worker), for the web site and
// the Android app: a second Python (Pyodide) that only thinks - the network's
// turns, the coach and "what the network thinks". The page keeps its own
// Python for the rules and stays smooth while the network thinks; it sends
// the game as save_state JSON, like it does to server.py on your own pc.
//
//   page -> worker: { id, kind: 'turn' | 'review' | 'view', model, body }
//   worker -> page: { id, out } (JSON text, as server.py answers) or { id, error }
//   page -> worker: { init: { models: [...] } } once, first

let py = null, ready = null;

async function boot(models) {
  importScripts('py/pyodide.js');
  const base = new URL('py/', self.location.href).href;
  // the standard library ships as base64 text (see boot() in index.html)
  const b64 = await (await fetch(base + 'python_stdlib.b64.txt')).text();
  const bin = Uint8Array.from(atob(b64.trim()), ch => ch.charCodeAt(0));
  const stdLibURL = base + 'python_stdlib_inline.zip';
  const realFetch = self.fetch.bind(self);
  self.fetch = (u, o) => String(u && u.url || u) === stdLibURL
    ? Promise.resolve(new Response(bin, { headers: { 'Content-Type': 'application/zip' } }))
    : realFetch(u, o);
  py = await loadPyodide({ indexURL: base, stdLibURL });
  self.fetch = realFetch;
  // numpy: a copy next to the page (py/, the app) if there is one, else from Pyodide's site
  const lock = await (await fetch(base + 'pyodide-lock.json')).json();
  const wheel = lock.packages.numpy.file_name;
  const here = await fetch(base + wheel, { method: 'HEAD' }).catch(() => null);
  await py.loadPackage(here && here.ok ? 'numpy'
    : 'https://cdn.jsdelivr.net/pyodide/v' + lock.info.version + '/full/' + wheel);
  const home = '/home/pyodide/';
  const put = async (url, path, binary) => {
    let r = await fetch(url, { cache: 'no-store' });
    if (!r.ok && url === 'looot.py') r = await fetch('looot.py.txt', { cache: 'no-store' });
    if (!r.ok) throw new Error('Could not load ' + url);
    py.FS.writeFile(path, binary ? new Uint8Array(await r.arrayBuffer()) : await r.text());
  };
  for (const f of ['looot.py', 'nn_encode.py', 'mp_encode.py', 'nn_bot.py', 'review.py', 'endgame.py'])
    await put(f, home + f);
  py.FS.mkdirTree(home + 'models');
  for (const m of models) await put('models/' + m + '.npz', home + 'models/' + m + '.npz', true);
  py.runPython(`
import json, random
import looot as L, nn_bot, review

def handle(kind, model, body):
    if kind == "turn":                       # the network plays the current turn
        g = L.Game.load_state(json.loads(body))
        if not g.game_over and g.player().ai:
            # its last 2 turns worked out exactly (narrower than on the pc)
            nn_bot.NNBot(random.Random(), model, endgame="narrow").play_turn(g)
        return json.dumps(g.save_state())
    if kind == "review":                     # the coach's judgement of one turn
        req = json.loads(body)
        return json.dumps(review.review_turn(model, req["before"], req["after"],
                                             search=bool(req.get("search", True)), endgame_width="narrow"))
    if kind == "view":                       # what the network thinks, from its own seat
        g = L.Game.load_state(json.loads(body))
        seat = next((p.idx for p in g.players if getattr(p, "nn", False)), None)
        if seat is None:
            return json.dumps({"error": "No network player in this game."})
        out = nn_bot.thoughts(nn_bot.load_net(model), g, seat)
        out["model"] = model
        return json.dumps(out)
    raise ValueError("unknown question: " + kind)
`);
  return py.globals.get('handle');
}

onmessage = async e => {
  const m = e.data;
  if (m.init) {
    ready = boot(m.init.models);
    ready.then(() => postMessage({ ready: true }), err => postMessage({ ready: false, error: String(err && err.message || err) }));
    return;
  }
  try {
    const handle = await ready;
    postMessage({ id: m.id, out: handle(m.kind, m.model, m.body) });
  } catch (err) {
    postMessage({ id: m.id, error: String(err && err.message || err) });
  }
};
