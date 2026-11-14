// Browser-only build (GitHub Pages): there is no Python server, so the page's /api/* calls are
// answered here. Lyrics come straight from LRCLIB (same lookup as lyrics.py, without the disk
// cache); "now playing" and vocal recognition (Whisper) need the desktop server and report unavailable.
(() => {
  if (!window.CHORDSTREAM_LOCAL) return;
  const API = 'https://lrclib.net/api';
  const NOISE = /\s*[([](official|lyric|lyrics|audio|video|hd|4k|remaster|visualizer)[^)\]]*[)\]]/gi;
  const realFetch = window.fetch.bind(window);
  const json = (obj) => new Response(JSON.stringify(obj), { headers: { 'Content-Type': 'application/json' } });

  async function get(path, params = {}) {
    const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v));
    const r = await realFetch(`${API}/${path}${q.toString() ? `?${q}` : ''}`);
    if (r.status === 400 || r.status === 404) return null;
    if (!r.ok) throw new Error(`LRCLIB HTTP ${r.status}`);
    return r.json();
  }
  function parseLrc(text) {
    const out = [];
    for (const raw of (text || '').split(/\r?\n/)) {
      const m = raw.trim().match(/^\[(\d+):(\d+(?:\.\d+)?)\](.*)$/);
      if (m) out.push({ t: Number(m[1]) * 60 + Number(m[2]), text: m[3].trim() });
    }
    return out.sort((a, b) => a.t - b.t);
  }
  const record = (r) => ({
    id: r.id, artist: r.artistName, title: r.trackName, album: r.albumName, duration: r.duration,
    synced: parseLrc(r.syncedLyrics), plain: r.plainLyrics || '', instrumental: !!r.instrumental,
  });
  async function find(artist, title, duration) {
    title = title.replace(NOISE, '').trim();
    const exact = await get('get', { artist_name: artist, track_name: title, duration: duration ? Math.round(duration) : null });
    if (exact && exact.syncedLyrics) return record(exact);
    let results = (await get('search', { artist_name: artist, track_name: title })) || [];
    if (!results.length) results = (await get('search', { q: `${artist} ${title}` })) || [];
    const synced = results.filter((r) => r.syncedLyrics), pool = synced.length ? synced : results;
    if (!pool.length) return null;
    const d = Number(duration) || 0;
    return record(d ? pool.reduce((a, b) => (Math.abs((b.duration || 0) - d) < Math.abs((a.duration || 0) - d) ? b : a)) : pool[0]);
  }

  window.fetch = async (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    const p = url.pathname.replace(/^.*\/api\//, '/api/');
    if (!url.pathname.includes('/api/') || url.origin !== location.origin) return realFetch(input, init);
    const q = url.searchParams;
    if (p === '/api/nowplaying') return json({ available: false, reason: 'browser-only version' });
    if (p === '/api/asr') return json({ enabled: false });
    if (p === '/api/lyrics/search') return json(((await get('search', { q: q.get('q') })) || []).slice(0, 10).map(record));
    if (p === '/api/lyrics') {
      if (q.get('id')) { const r = await get(`get/${Number(q.get('id'))}`); return json(r ? record(r) : { synced: [], plain: '', missing: true }); }
      return json((await find(q.get('artist') || '', q.get('title') || '', q.get('duration'))) || { synced: [], plain: '', missing: true });
    }
    return new Response('not available in the browser-only version', { status: 404 });
  };
})();
