// Time-synced lyrics from LRCLIB (https://lrclib.net): free, no key, CORS-enabled, so the page
// calls it directly. window.lrclib = {find(artist, title, duration), search(q), byId(id)}; records
// are {id, artist, title, album, duration, synced: [{t, text}], plain, instrumental}.
(() => {
  const API = 'https://lrclib.net/api';
  // YouTube-style title noise that hurts matching.
  const NOISE = /\s*[([](official|lyric|lyrics|audio|video|hd|4k|remaster|visualizer)[^)\]]*[)\]]/gi;

  async function get(path, params = {}) {
    const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v));
    const r = await fetch(`${API}/${path}${q.toString() ? `?${q}` : ''}`);
    if (r.status === 400 || r.status === 404) return null;  // 400: e.g. no artist (files named by title only)
    if (!r.ok) throw new Error(`LRCLIB HTTP ${r.status}`);
    return r.json();
  }
  const cleanTitle = (title) => title.replace(NOISE, '').trim();
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
  // Exact match first; otherwise search and prefer synced lyrics whose length matches.
  async function find(artist, title, duration) {
    title = cleanTitle(title);
    const exact = await get('get', { artist_name: artist, track_name: title, duration: duration ? Math.round(duration) : null });
    if (exact && exact.syncedLyrics) return record(exact);
    let results = (await get('search', { artist_name: artist, track_name: title })) || [];
    if (!results.length) results = (await get('search', { q: `${artist} ${title}` })) || [];
    const synced = results.filter((r) => r.syncedLyrics), pool = synced.length ? synced : results;
    if (!pool.length) return null;
    const d = Number(duration) || 0;
    return record(d ? pool.reduce((a, b) => (Math.abs((b.duration || 0) - d) < Math.abs((a.duration || 0) - d) ? b : a)) : pool[0]);
  }

  async function search(q) { return ((await get('search', { q })) || []).slice(0, 10).map(record); }
  async function byId(id) { const r = await get(`get/${Number(id)}`); return r ? record(r) : null; }
  window.lrclib = { find, search, byId, parseLrc, cleanTitle };
})();
