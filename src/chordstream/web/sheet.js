// Synced chord sheet: lyrics from LRCLIB, song + position from the OS media session,
// chords placed above the word being sung when they were detected.
//
// Word timing: when the server runs Whisper (serve --asr), recognised sung words are aligned
// to the lyric words (fuzzy, time-constrained DP). Aligned words become anchors; the other
// words are interpolated between anchors. Line-initial anchors also auto-correct the lyrics
// offset. Without ASR, words are timed by character position within the line.
(() => {
  const $ = (id) => document.getElementById(id);
  const LAG = 0.12;          // capture + transport + model emission delay (s)
  const POLL_MS = 1500;

  const st = {
    np: null, npAt: 0,       // last /api/nowplaying response and when it arrived (performance.now)
    key: null, meta: null,   // current song identity
    lines: [],               // [{t, text}]
    chords: [],              // [{t (playback seconds), name, label}]
    offset: 0,               // lyrics time = playback time + offset
    anchor: null,            // manual clock {pos, perf} when the player exposes no position
    followUntil: 0,          // pause auto-scroll after the user scrolls
    lastNow: -1,
    clock: null,             // {streamT, perf}: maps server stream time to performance.now()
    ws: null, lastPrompt: null,
    tokens: [],              // per line: [{text, norm, c0, c1}] (c0/c1 = glyph offsets)
    heard: [],               // recognised words [{t (playback s), norm, w, p}]
    anchors: new Map(),      // lineIdx -> Map(wordIdx -> playback time)
    res: [],                 // recent line-start residuals for auto offset
    auto: true, asr: null,
  };
  const RATE = 0.085;        // fallback singing speed, seconds per character

  // ------------------------------------------------------------ clocks
  function playbackTime(at = performance.now()) {
    const np = st.np;
    if (np && np.has_timeline && np.position != null) return np.position + (np.playing ? (at - st.npAt) / 1000 : 0);
    if (st.anchor) return st.anchor.pos + (at - st.anchor.perf) / 1000;
    return null;
  }
  const lyricTime = (pt) => (pt == null ? null : pt + st.offset);
  // Server stream time (seconds of audio received) -> playback time of that audio.
  function streamToPlayback(ts) {
    if (!st.clock) return null;
    const pt = playbackTime(st.clock.perf - (st.clock.streamT - ts) * 1000);
    return pt == null ? null : pt - LAG;
  }

  // ------------------------------------------------------------ now playing + lyrics
  async function poll() {
    try {
      const np = await (await fetch('/api/nowplaying')).json();
      st.np = np; st.npAt = performance.now();
      if (np.available && np.title) {
        const key = `${np.artist}|${np.title}`.toLowerCase();
        if (key !== st.key) newSong(key, np);
      }
      renderHeader();
    } catch { /* server restarting: try again next tick */ }
  }

  async function newSong(key, np) {
    st.key = key; st.meta = { artist: np.artist, title: np.title };
    st.chords = []; st.lines = []; st.anchor = null; st.offset = 0;
    st.heard = []; st.anchors = new Map(); st.res = []; st.tokens = []; st.lastPrompt = null;
    setOffsetLabel();
    $('sheet').innerHTML = '<p class="sheet-empty">Looking up lyrics…</p>';
    const q = new URLSearchParams({ artist: np.artist, title: np.title });
    if (np.duration) q.set('duration', Math.round(np.duration));
    try { loadLyrics(await (await fetch(`/api/lyrics?${q}`)).json()); }
    catch { $('sheet').innerHTML = '<p class="sheet-empty">Lyrics lookup failed. Search manually above.</p>'; }
  }

  function loadLyrics(rec) {
    st.lines = (rec.synced || []).filter((l) => l.t != null);
    st.tokens = st.lines.map((l) => tokenize(l.text));
    st.anchors = new Map(); st.res = []; st.lastPrompt = null;
    if (rec.title) st.meta = { ...(st.meta || {}), lyricsTitle: `${rec.artist} – ${rec.title}` };
    if (!st.lines.length && rec.plain) {
      st.lines = []; st.plain = rec.plain;
    } else st.plain = '';
    renderHeader(); render();
  }

  // ------------------------------------------------------------ chords in
  function addChord(name, label, ageSeconds) {
    const pt = playbackTime();
    if (pt == null) return;
    const t = pt - ageSeconds - LAG;
    // A jump backwards means the user seeked: forget what was recorded after that point.
    const last = st.chords[st.chords.length - 1];
    if (last && t < last.t - 2) st.chords = st.chords.filter((c) => c.t < t - 0.25);
    if (last && last.name === name && Math.abs(last.t - t) < 0.5) return;
    st.chords.push({ t, name, label });
    render();
  }

  // ------------------------------------------------------------ layout
  const segmenter = typeof Intl !== 'undefined' && Intl.Segmenter ? new Intl.Segmenter(undefined, { granularity: 'grapheme' }) : null;
  const glyphs = (s) => (segmenter ? [...segmenter.segment(s)].length : [...s].length);

  // Phonetic key: lowercase, strip Latin diacritics (ā -> a), keep letters/marks/digits, then
  // fold common romanization variants (aa/ā, ee/ī, th/t, w/v, ai/ei, final nasal, doubled letters)
  // so "Aahatein"/"āhaṭeṃ" and "theri"/"terī" compare equal. Non-Latin scripts keep their marks.
  const norm = (w) => {
    let k = w.toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/[^\p{L}\p{M}\p{N}]/gu, '');
    if (!/^[a-z0-9]+$/.test(k)) return k;
    k = k.replace(/ee/g, 'i').replace(/oo/g, 'u').replace(/([kgcjtdpbrs])h/g, '$1').replace(/ph/g, 'f')
      .replace(/w/g, 'v').replace(/z/g, 'j').replace(/q/g, 'k').replace(/[ae]i/g, 'e').replace(/[ao]u/g, 'o')
      .replace(/(.)\1+/g, '$1');
    return k.length > 3 ? k.replace(/([aeiou])[mn]$/, '$1') : k;
  };
  function tokenize(text) {
    let c = 0;
    return text.split(/\s+/).filter(Boolean).map((w) => {
      const g = glyphs(w), tok = { text: w, norm: norm(w), c0: c, c1: c + g };
      c += g + 1;
      return tok;
    });
  }
  const nextT = (i) => (i + 1 < st.lines.length ? st.lines[i + 1].t : Infinity);

  // Lyric-time start of every word in line i: anchors where heard, interpolation elsewhere.
  function wordTimes(i) {
    const line = st.lines[i], toks = st.tokens[i] || [];
    if (!toks.length) return [];
    const n = toks[toks.length - 1].c1;
    const anchors = st.anchors.get(i);
    const known = anchors ? [...anchors.entries()].map(([k, pb]) => [toks[k].c0, pb + st.offset]).sort((a, b) => a[0] - b[0]) : [];
    const limit = nextT(i);
    let rate = Math.min(RATE, Math.max(0.03, (Math.min(limit, line.t + n * RATE + 0.4) - line.t) / Math.max(1, n)));
    if (known.length >= 2) {  // measured singing speed for this line
      const a = known[0], b = known[known.length - 1];
      if (b[0] > a[0]) rate = Math.max(0.03, Math.min(0.4, (b[1] - a[1]) / (b[0] - a[0])));
    }
    if (!known.length) return toks.map((t) => line.t + t.c0 * rate);
    const out = toks.map((t) => {
      let lo = null, hi = null;
      for (const kp of known) { if (kp[0] <= t.c0) lo = kp; if (kp[0] >= t.c0 && !hi) hi = kp; }
      if (lo && hi && hi[0] !== lo[0]) return lo[1] + (hi[1] - lo[1]) * (t.c0 - lo[0]) / (hi[0] - lo[0]);
      if (lo && hi) return lo[1];
      if (lo) return lo[1] + (t.c0 - lo[0]) * rate;
      return Math.max(line.t - 1, hi[1] - (hi[0] - t.c0) * rate);
    });
    for (let k = 1; k < out.length; k++) out[k] = Math.max(out[k], out[k - 1]);  // keep monotonic
    return out;
  }

  function similarity(a, b) {
    if (!a || !b) return 0;
    if (a === b) return 1;
    if (Math.abs(a.length - b.length) > Math.max(a.length, b.length) / 2) return 0;
    const d = Array.from({ length: b.length + 1 }, (_, j) => j);
    for (let i = 1; i <= a.length; i++) {
      let prev = d[0]; d[0] = i;
      for (let j = 1; j <= b.length; j++) {
        const tmp = d[j];
        d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
        prev = tmp;
      }
    }
    const r = 1 - d[b.length] / Math.max(a.length, b.length);
    return r >= 0.7 ? r * 0.8 : 0;
  }

  // Align recognised words to lyric words near the current position (monotonic DP).
  function align() {
    const lt = lyricTime(playbackTime());
    if (lt == null || !st.lines.length) return;
    const A = [];
    st.lines.forEach((line, i) => {
      if (!line.text || line.t < lt - 25 || line.t > lt + 6) return;
      const wt = wordTimes(i);
      st.tokens[i].forEach((tok, k) => A.push({ i, k, norm: tok.norm, est: wt[k] }));
    });
    if (!A.length) return;
    const lo = A[0].est - 4, hi = A[A.length - 1].est + 4;
    const B = st.heard.filter((h) => h.norm && h.t + st.offset >= lo && h.t + st.offset <= hi);
    if (!B.length) return;
    const n = A.length, m = B.length;
    const score = (a, b) => (Math.abs(b.t + st.offset - a.est) <= 5 ? similarity(a.norm, b.norm) : 0);
    const SC = Array.from({ length: n }, (_, x) => Float32Array.from(B, (b) => score(A[x], b)));
    const S = Array.from({ length: n + 1 }, () => new Float32Array(m + 1));
    for (let x = 1; x <= n; x++) for (let y = 1; y <= m; y++) {
      const sc = SC[x - 1][y - 1];
      S[x][y] = Math.max(S[x - 1][y], S[x][y - 1], sc > 0 ? S[x - 1][y - 1] + sc : -1);
    }
    const matches = [];
    for (let x = n, y = m; x > 0 && y > 0;) {
      const sc = SC[x - 1][y - 1];
      if (sc > 0 && S[x][y] === S[x - 1][y - 1] + sc) { matches.push([A[x - 1], B[y - 1]]); x--; y--; }
      else if (S[x][y] === S[x - 1][y]) x--;
      else y--;
    }
    for (const [a, b] of matches) {
      if (!st.anchors.has(a.i)) st.anchors.set(a.i, new Map());
      st.anchors.get(a.i).set(a.k, b.t);
      if (a.k === 0 && b.p >= 0.5) st.res.push(b.t + st.offset - st.lines[a.i].t);
    }
    st.res = st.res.slice(-9);
    // Auto offset: a line should start when its first word is actually sung.
    if (st.auto && st.res.length >= 3) {
      const med = [...st.res].sort((p, q) => p - q)[st.res.length >> 1];
      if (Math.abs(med) > 0.15) {
        const shift = -med * 0.6;
        st.offset += shift;
        st.res = st.res.map((r) => r + shift);
        setOffsetLabel();
      }
    }
  }

  function onWords(msg) {
    if (!msg.words) return;
    const lo = streamToPlayback(msg.t0 + 0.3), hi = streamToPlayback(msg.t1 - 0.6);
    if (lo == null || hi == null) return;
    const fresh = msg.words
      .map((w) => ({ t: streamToPlayback(w.start), w: w.w, norm: norm(w.rom || w.w), p: w.p }))
      .filter((w) => w.t != null && w.t >= lo && w.t <= hi);
    st.heard = st.heard.filter((h) => h.t < lo || h.t > hi).concat(fresh).sort((a, b) => a.t - b.t).slice(-500);
    align();
    render();
    renderHeader();
  }
  function onEvent(e) { st.clock = { streamT: e.time, perf: performance.now() }; }
  function attachSocket(ws) { st.ws = ws; st.lastPrompt = null; }

  // Tell the server which lines come next: Whisper uses them as its prompt.
  function sendContext() {
    if (!st.ws || st.ws.readyState !== 1 || !st.lines.length) return;
    const lt = lyricTime(playbackTime());
    if (lt == null) return;
    let cur = 0;
    st.lines.forEach((l, i) => { if (l.t <= lt) cur = i; });
    const prompt = st.lines.slice(Math.max(0, cur - 1), cur + 4).map((l) => l.text).filter(Boolean).join(' ');
    if (prompt && prompt !== st.lastPrompt) {
      // Latin-script lyrics: ask the server to romanize what it hears (e.g. Devanagari -> Latin).
      const latin = (prompt.match(/[a-z]/gi) || []).length > prompt.replace(/\s/g, '').length * 0.6;
      st.ws.send(JSON.stringify({ type: 'context', prompt, latin, song: st.key }));
      st.lastPrompt = prompt;
    }
  }

  // Turn lines + chords into blocks: {kind: 'line'|'inst', t, end, words:[{text, chords:[]}], chords:[]}
  function layout() {
    const chords = st.chords.map((c) => ({ ...c, lt: lyricTime(c.t) })).sort((a, b) => a.lt - b.lt);
    const blocks = [];
    const lines = st.lines;
    let ci = 0;
    const take = (until) => { const out = []; while (ci < chords.length && chords[ci].lt < until) out.push(chords[ci++]); return out; };

    if (!lines.length) {
      blocks.push({ kind: 'inst', label: 'Chords', chords: take(Infinity) });
      return blocks;
    }
    const intro = take(lines[0].t);
    if (intro.length) blocks.push({ kind: 'inst', label: 'Intro', chords: intro, t: 0, end: lines[0].t });

    lines.forEach((line, i) => {
      const next = i + 1 < lines.length ? lines[i + 1].t : Infinity;
      const inLine = take(next);
      if (!line.text) { // LRC blank line = instrumental break
        blocks.push({ kind: 'inst', label: 'Instrumental', chords: inLine, t: line.t, end: next });
        return;
      }
      const toks = st.tokens[i] || [];
      const anchors = st.anchors.get(i);
      const words = toks.map((t, k) => ({ text: t.text, chords: [], heard: !!(anchors && anchors.has(k)) }));
      const wt = wordTimes(i);
      const lastTok = toks[toks.length - 1];
      const sungEnd = Math.min(next, wt[wt.length - 1] + Math.max(0.4, (lastTok.c1 - lastTok.c0) * RATE));
      const tail = [];
      for (const c of inLine) {
        if (c.lt > sungEnd + 0.8) { tail.push(c); continue; }
        // The chord goes on the word being sung at that moment (chords often land just before the syllable).
        let wi = 0;
        while (wi + 1 < wt.length && wt[wi + 1] <= c.lt + 0.12) wi++;
        words[wi].chords.push(c);
      }
      blocks.push({ kind: 'line', t: line.t, end: next, words });
      if (tail.length) blocks.push({ kind: 'inst', label: '', chords: tail, t: sungEnd, end: next });
    });
    return blocks;
  }

  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const chordHtml = (cs) => cs.map((c) => `<b data-label="${esc(c.label)}">${esc(c.name)}</b>`).join(' ');

  function render() {
    const sheet = $('sheet');
    if (!st.lines.length && st.plain) {
      sheet.innerHTML = `<p class="sheet-empty">Only unsynced lyrics were found, so chords can't be placed on words.</p>
        <div class="block inst"><div class="chordrow">${chordHtml(st.chords)}</div></div>
        <pre class="plain">${esc(st.plain)}</pre>`;
      return;
    }
    if (!st.lines.length && !st.chords.length) {
      sheet.innerHTML = `<p class="sheet-empty">${st.key ? 'No synced lyrics found for this song. Try the search box.' : 'Play a song in any app. It will show up here.'}</p>`;
      return;
    }
    sheet.innerHTML = layout().map((b, i) => {
      const attrs = `data-i="${i}" data-t="${b.t ?? ''}" data-end="${b.end ?? ''}"`;
      if (b.kind === 'inst') {
        if (!b.chords.length && !b.label) return '';
        return `<div class="block inst" ${attrs}>${b.label ? `<span class="tag">${b.label}</span>` : ''}<div class="chordrow">${chordHtml(b.chords) || '<i>…</i>'}</div></div>`;
      }
      const words = b.words.map((w) => `<span class="w${w.heard ? ' heard' : ''}"><span class="c">${chordHtml(w.chords) || '&nbsp;'}</span>${esc(w.text)}</span>`).join(' ');
      return `<div class="block line" ${attrs}>${words}</div>`;
    }).join('');
    st.lastNow = -1;
    highlight();
  }

  function highlight() {
    const lt = lyricTime(playbackTime());
    const blocks = [...document.querySelectorAll('#sheet .block[data-t]')];
    let now = -1;
    blocks.forEach((b, i) => {
      const t = parseFloat(b.dataset.t), end = parseFloat(b.dataset.end || 'Infinity');
      b.classList.toggle('past', lt != null && end <= lt);
      if (lt != null && t <= lt && lt < end) now = i;
    });
    if (now !== st.lastNow) {
      blocks.forEach((b, i) => b.classList.toggle('now', i === now));
      st.lastNow = now;
      if (now >= 0 && performance.now() > st.followUntil) {
        // Position relative to the scroll box itself (offsetTop is relative to the card, which overshoots).
        const box = $('sheet'), el = blocks[now];
        const top = el.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
        box.scrollTo({ top: Math.max(0, top - 8), behavior: 'smooth' });
      }
    }
  }

  // ------------------------------------------------------------ header + controls
  function fmt(t) { if (t == null) return '–'; const s = Math.max(0, t); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`; }
  function renderHeader() {
    const np = st.np;
    const title = st.meta ? st.meta.title : (np && np.reason) || 'Nothing playing';
    $('songtitle').textContent = title || 'Nothing playing';
    $('songartist').textContent = st.meta ? st.meta.artist : '';
    const pt = playbackTime();
    let sync;
    if (np && np.has_timeline) sync = `${np.playing ? 'Playing' : 'Paused'} · ${fmt(pt)} / ${fmt(np.duration)} · synced to ${appName(np.app)}`;
    else if (st.anchor) sync = `Manual sync · ${fmt(pt)}`;
    else if (np && np.available) sync = `${appName(np.app)} doesn't report position. Click the line being sung to sync`;
    else sync = np && np.reason ? np.reason : '';
    $('songsync').textContent = sync;
    const src = st.meta && st.meta.lyricsTitle ? `Lyrics: LRCLIB (${st.meta.lyricsTitle})` : '';
    let vs = '';
    if (st.asr && st.asr.enabled) {
      const n = [...st.anchors.values()].reduce((a, m) => a + m.size, 0);
      vs = st.asr.ready ? `Vocal sync (Whisper ${st.asr.model} on ${st.asr.device}): ${n} words matched`
        : st.asr.error ? `Vocal sync unavailable: ${st.asr.error}` : 'Vocal sync: loading model…';
    } else if (st.asr) vs = 'Vocal sync off (start the server with --asr small)';
    $('songsrc').textContent = [src, vs].filter(Boolean).join(' · ');
  }
  function appName(id = '') {
    const low = id.toLowerCase();
    if (low.includes('spotify')) return 'Spotify';
    if (low.includes('youtube-music')) return 'YouTube Music';
    if (low.includes('chrome')) return 'Chrome';
    if (low.includes('edge')) return 'Edge';
    if (low.includes('firefox') || /^[0-9A-F]{16}$/.test(id)) return 'Firefox';
    return id.split(/[\\!.]/).filter(Boolean).pop() || 'player';
  }
  function setOffsetLabel() { $('offset').textContent = `${st.offset >= 0 ? '+' : ''}${st.offset.toFixed(1)} s`; }

  function bindControls() {
    $('offminus').onclick = () => { st.offset -= 0.1; st.res = []; setOffsetLabel(); render(); };
    $('offplus').onclick = () => { st.offset += 0.1; st.res = []; setOffsetLabel(); render(); };
    $('autosync').onchange = (ev) => { st.auto = ev.target.checked; st.res = []; };
    // Only real user input pauses auto-follow (programmatic smooth scrolling also fires 'scroll').
    const hold = () => { st.followUntil = performance.now() + 4000; };
    for (const ev of ['wheel', 'touchmove', 'keydown', 'pointerdown']) $('sheet').addEventListener(ev, hold, { passive: true });
    // Click a lyric line at the moment it is sung: re-syncs lyrics to the music.
    $('sheet').addEventListener('click', (ev) => {
      const b = ev.target.closest('.block[data-t]');
      if (!b || b.dataset.t === '') return;
      const t = parseFloat(b.dataset.t);
      const pt = playbackTime();
      if (pt == null) { st.anchor = { pos: t, perf: performance.now() }; st.offset = 0; }
      else st.offset = t - pt;
      st.res = [];
      setOffsetLabel(); render(); renderHeader();
    });
    $('clearsheet').onclick = () => { st.chords = []; st.heard = []; st.anchors = new Map(); st.res = []; render(); };
    $('copysheet').onclick = async () => {
      try { await navigator.clipboard.writeText(toText()); flash($('copysheet'), 'Copied'); } catch { flash($('copysheet'), 'Blocked'); }
    };
    $('dlsheet').onclick = () => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(new Blob([toText()], { type: 'text/plain;charset=utf-8' }));
      a.download = `${(st.meta ? `${st.meta.artist} - ${st.meta.title}` : 'chords').replace(/[\\/:*?"<>|]/g, '_')}.txt`;
      a.click();
    };
    const form = $('lyricsearch');
    form.onsubmit = async (ev) => {
      ev.preventDefault();
      const q = $('lyricq').value.trim();
      if (!q) return;
      const box = $('results');
      box.innerHTML = '<span class="muted">Searching…</span>';
      try {
        const res = await (await fetch(`/api/lyrics/search?q=${encodeURIComponent(q)}`)).json();
        box.innerHTML = '';
        res.filter((r) => r.synced.length).slice(0, 6).forEach((r) => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.textContent = `${r.artist} – ${r.title} (${fmt(r.duration)})`;
          btn.onclick = () => { st.meta = { ...(st.meta || { artist: r.artist, title: r.title }) }; loadLyrics(r); box.innerHTML = ''; };
          box.append(btn);
        });
        if (!box.children.length) box.innerHTML = '<span class="muted">No synced lyrics found.</span>';
      } catch { box.innerHTML = '<span class="muted">Search failed.</span>'; }
    };
  }
  function flash(btn, text) { const old = btn.textContent; btn.textContent = text; setTimeout(() => (btn.textContent = old), 1200); }

  // Plain-text export in the familiar chords-over-lyrics layout.
  function toText() {
    const head = st.meta ? `${st.meta.title}\n${st.meta.artist}\n\n` : '';
    return head + layout().map((b) => {
      if (b.kind === 'inst') return b.chords.length ? `${b.label ? `[${b.label}]\n` : ''}${b.chords.map((c) => c.name).join('   ')}\n` : '';
      let top = '', bottom = '';
      b.words.forEach((w) => {
        const ch = w.chords.map((c) => c.name).join(' ');
        const width = Math.max(glyphs(w.text), ch.length) + 1;  // column width shared by chord and word
        top += ch.padEnd(width);
        bottom += w.text + ' '.repeat(width - glyphs(w.text));
      });
      return `${top.trimEnd()}\n${bottom.trimEnd()}\n`;
    }).join('\n');
  }

  // ------------------------------------------------------------ guitar chord diagrams
  const PCS = { C: 0, 'C#': 1, D: 2, Eb: 3, E: 4, F: 5, 'F#': 6, G: 7, Ab: 8, A: 9, Bb: 10, B: 11 };
  const OPEN = {
    'C:maj': [-1, 3, 2, 0, 1, 0], 'A:maj': [-1, 0, 2, 2, 2, 0], 'G:maj': [3, 2, 0, 0, 0, 3], 'E:maj': [0, 2, 2, 1, 0, 0],
    'D:maj': [-1, -1, 0, 2, 3, 2], 'A:min': [-1, 0, 2, 2, 1, 0], 'E:min': [0, 2, 2, 0, 0, 0], 'D:min': [-1, -1, 0, 2, 3, 1],
  };
  function shape(label) {
    if (OPEN[label]) return { frets: OPEN[label], barre: 0 };
    const [root, q] = label.split(':');
    const pc = PCS[root], minor = q === 'min';
    const fe = (pc - 4 + 12) % 12 || 12, fa = (pc - 9 + 12) % 12 || 12;
    if (fe <= fa) {
      const f = fe;
      return { frets: minor ? [f, f + 2, f + 2, f, f, f] : [f, f + 2, f + 2, f + 1, f, f], barre: f, from: 0 };
    }
    const f = fa;
    return { frets: minor ? [-1, f, f + 2, f + 2, f + 1, f] : [-1, f, f + 2, f + 2, f + 2, f], barre: f, from: 1 };
  }
  window.chordDiagram = (label) => {
    if (!label || !label.includes(':')) return '';
    const { frets, barre, from } = shape(label);
    const played = frets.filter((f) => f > 0);
    const base = Math.max(...played) > 4 ? Math.min(...played) : 1;
    const W = 132, H = 136, x0 = 30, y0 = 24, sx = 18, sy = 22;
    const X = (s) => x0 + s * sx, Y = (f) => y0 + (f - base + 0.5) * sy;
    let g = '';
    for (let f = 0; f <= 4; f++) g += `<line x1="${X(0)}" x2="${X(5)}" y1="${y0 + f * sy}" y2="${y0 + f * sy}" class="fret"/>`;
    for (let s = 0; s < 6; s++) g += `<line x1="${X(s)}" x2="${X(s)}" y1="${y0}" y2="${y0 + 4 * sy}" class="str"/>`;
    if (base === 1) g += `<rect x="${X(0) - 1}" y="${y0 - 4}" width="${5 * sx + 2}" height="4" rx="1" class="nut"/>`;
    else g += `<text x="${X(0) - 9}" y="${Y(base) + 4}" class="base">${base}</text>`;
    if (barre) g += `<line x1="${X(from)}" x2="${X(5)}" y1="${Y(barre)}" y2="${Y(barre)}" class="barre"/>`;
    frets.forEach((f, s) => {
      if (f < 0) g += `<text x="${X(s)}" y="${y0 - 8}" class="mark">×</text>`;
      else if (f === 0) g += `<circle cx="${X(s)}" cy="${y0 - 12}" r="4" class="open"/>`;
      else if (f !== barre) g += `<circle cx="${X(s)}" cy="${Y(f)}" r="6.5" class="dot"/>`;
    });
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Guitar chord ${label}">${g}</svg>`;
  };

  // ------------------------------------------------------------ boot
  window.sheet = { addChord, onWords, onEvent, attachSocket, debug: () => st };
  bindControls();
  poll();
  const pollAsr = async () => { try { st.asr = await (await fetch('/api/asr')).json(); renderHeader(); } catch {} };
  pollAsr();
  setInterval(() => { if (!st.asr || (st.asr.enabled && !st.asr.ready && !st.asr.error)) pollAsr(); }, 3000);
  setInterval(sendContext, 1000);
  setInterval(() => { if (!document.hidden) poll(); }, POLL_MS);
  setInterval(() => { highlight(); if (st.np && st.np.playing) renderHeader(); }, 250);
})();
