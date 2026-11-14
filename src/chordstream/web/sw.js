// Rubato service worker: keeps the page, engine, onnxruntime and model (~14 MB) in a cache named
// after the build, so repeat visits load at once and work offline. The page itself is
// network-first (a new deploy is picked up when online); everything else on this origin is
// cache-first. Lyrics (LRCLIB) are never cached. scripts/build_site.py sets VERSION.
const VERSION = 'dev';
const CACHE = `rubato-${VERSION}`;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (ev) => {
  ev.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith('rubato-') && k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});
self.addEventListener('fetch', (ev) => {
  const req = ev.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return;
  if (req.mode === 'navigate') {
    ev.respondWith((async () => {
      try {
        const res = await fetch(req);
        const c = await caches.open(CACHE); c.put(req, res.clone());
        return res;
      } catch { return (await caches.match(req)) || Response.error(); }
    })());
    return;
  }
  ev.respondWith((async () => {
    const hit = await caches.match(req);
    if (hit) return hit;
    const res = await fetch(req);
    if (res.ok && res.status === 200) { const c = await caches.open(CACHE); c.put(req, res.clone()); }
    return res;
  })());
});
