// The service worker of the Looot web app: once the game has been opened,
// it also works without internet, and it starts faster.
//
//  - Big files that don't change (Pyodide in py/, the networks in models/,
//    numpy from Pyodide's site, the fonts, the icons): from the phone's own
//    copy first, the internet only the first time.
//  - Everything that can change (the pages, the rules in looot.py, ...):
//    from the internet first, so a new version arrives at once; the copy
//    only when there is no internet.
//
// index.html asks it to fetch the networks and numpy in advance ("warm"),
// so playing against the network also works offline after the first visit.

const CACHE = 'looot-v1';
const CORE = [
  './', 'index.html', 'manifest.webmanifest', 'boards.json',
  'looot.py', 'nn_bot.py', 'nn_encode.py', 'mp_encode.py', 'review.py', 'endgame.py',
  'py/pyodide.js', 'py/pyodide.asm.js', 'py/pyodide.asm.wasm',
  'py/python_stdlib.b64.txt', 'py/pyodide-lock.json',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/favicon-32.png',
];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(CORE)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  // older versions of the copy are removed
  e.waitUntil(caches.keys()
    .then(keys => Promise.all(keys.filter(k => k.startsWith('looot-') && k !== CACHE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});

const STATIC = /\/(py|models|icons)\//;
const FOREIGN = /(^|\.)(cdn\.jsdelivr\.net|fonts\.googleapis\.com|fonts\.gstatic\.com)$/;

async function cacheFirst(req) {
  const hit = await caches.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok || res.type === 'opaque') (await caches.open(CACHE)).put(req, res.clone());
  return res;
}

async function networkFirst(req) {
  try {
    const res = await fetch(req);
    if (res.ok) (await caches.open(CACHE)).put(req, res.clone());
    return res;
  } catch (err) {
    const hit = await caches.match(req, { ignoreSearch: true });
    if (hit) return hit;
    if (req.mode === 'navigate') {
      const page = await caches.match('index.html');
      if (page) return page;
    }
    throw err;
  }
}

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;                  // the local server's moves etc.
  const url = new URL(req.url);
  if (url.origin === location.origin) {
    e.respondWith(STATIC.test(url.pathname) ? cacheFirst(req) : networkFirst(req));
  } else if (FOREIGN.test(url.hostname)) {
    e.respondWith(cacheFirst(req));
  }
});

// The page asks for files it will need later (the networks, numpy).
self.addEventListener('message', e => {
  const d = e.data || {};
  if (d.type !== 'warm' || !Array.isArray(d.urls)) return;
  e.waitUntil((async () => {
    const c = await caches.open(CACHE);
    for (const u of d.urls) {
      try {
        if (!(await c.match(u))) {
          const res = await fetch(u);
          if (res.ok) await c.put(u, res);
        }
      } catch (err) { /* offline: next time */ }
    }
  })());
});
