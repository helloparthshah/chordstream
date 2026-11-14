// Synced chord sheet: time-synced lyrics from LRCLIB (lyrics.js), chords placed above the word being
// sung when they were detected. Song + clock: a file played in the page ("Artist - Title.mp3"), or
// a lyrics search plus a click on the line being sung (manual sync). Words are timed by character
// position within the line.
(() => {
  const $ = (id) => document.getElementById(id);
  const LAG = 0.12;          // capture + model emission delay (s)

  const st = {
    key: null, meta: null,   // current song identity
    lines: [],               // [{t, text}]
    chords: [],              // [{t (playback seconds), name, label}]
    offset: 0,               // lyrics time = playback time + offset
    anchor: null,            // manual clock {pos, perf}: set by clicking the line being sung
    followUntil: 0,          // pause auto-scroll after the user scrolls
    lastNow: -1,
    tokens: [],              // per line: [{text, norm, c0, c1}] (c0/c1 = glyph offsets)
    local: null,             // playing a local file: {name, time(), playing(), duration()}
  };
  const RATE = 0.085;        // fallback singing speed, seconds per character

  // ------------------------------------------------------------ clocks
  function playbackTime(at = performance.now()) {
    if (st.local) {  // local file: the <audio> element's own clock (exact, follows seeks)
      const t = st.local.time();
      return st.local.playing() ? t + (at - performance.now()) / 1000 : t;
    }
    if (st.anchor) return st.anchor.pos + (at - st.anchor.perf) / 1000;
    return null;
  }
  const lyricTime = (pt) => (pt == null ? null : pt + st.offset);

  // ------------------------------------------------------------ lyrics
  async function newSong(key, np) {
    st.key = key; st.meta = { artist: np.artist, title: np.title };
    st.chords = []; st.lines = []; st.anchor = null; st.offset = 0;
    st.tokens = [];
    setOffsetLabel();
    $('sheet').innerHTML = '<p class="sheet-empty">Looking up lyrics…</p>';
    try { loadLyrics((await window.lrclib.find(np.artist, np.title, np.duration)) || { synced: [], plain: '', missing: true }); }
    catch { $('sheet').innerHTML = '<p class="sheet-empty">Lyrics lookup failed. Search manually above.</p>'; }
  }

  function loadLyrics(rec) {
    st.lines = (rec.synced || []).filter((l) => l.t != null);
    st.tokens = st.lines.map((l) => tokenize(l.text));
    if (rec.title) st.meta = { ...(st.meta || {}), lyricsTitle: `${rec.artist} – ${rec.title}` };
    if (!st.lines.length && rec.plain) {
      st.lines = []; st.plain = rec.plain;
    } else st.plain = '';
    renderHeader(); render();
  }

  // ------------------------------------------------------------ chords in
  function addChord(name, label, ageSeconds) {
    if (st.local && !st.local.playing()) return;  // paused: the engine only hears silence
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

  function tokenize(text) {
    let c = 0;
    return text.split(/\s+/).filter(Boolean).map((w) => {
      const g = glyphs(w), tok = { text: w, c0: c, c1: c + g };
      c += g + 1;
      return tok;
    });
  }
  const nextT = (i) => (i + 1 < st.lines.length ? st.lines[i + 1].t : Infinity);

  // Lyric-time start of every word in line i, from its character position within the line.
  function wordTimes(i) {
    const line = st.lines[i], toks = st.tokens[i] || [];
    if (!toks.length) return [];
    const n = toks[toks.length - 1].c1;
    const rate = Math.min(RATE, Math.max(0.03, (Math.min(nextT(i), line.t + n * RATE + 0.4) - line.t) / Math.max(1, n)));
    return toks.map((t) => line.t + t.c0 * rate);
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
      const words = toks.map((t) => ({ text: t.text, chords: [] }));
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
      sheet.innerHTML = `<p class="sheet-empty">${st.key ? 'No synced lyrics found for this song. Try the search box.' : 'Search for the lyrics above, or play a file named "Artist - Title". Chords appear over the words as they are sung.'}</p>`;
      return;
    }
    sheet.innerHTML = layout().map((b, i) => {
      const attrs = `data-i="${i}" data-t="${b.t ?? ''}" data-end="${b.end ?? ''}"`;
      if (b.kind === 'inst') {
        if (!b.chords.length && !b.label) return '';
        return `<div class="block inst" ${attrs}>${b.label ? `<span class="tag">${b.label}</span>` : ''}<div class="chordrow">${chordHtml(b.chords) || '<i>…</i>'}</div></div>`;
      }
      const words = b.words.map((w) => `<span class="w"><span class="c">${chordHtml(w.chords) || '&nbsp;'}</span>${esc(w.text)}</span>`).join(' ');
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
    if (st.local) {
      $('songtitle').textContent = st.meta ? st.meta.title : st.local.name;
      $('songartist').textContent = st.meta ? st.meta.artist : '';
      const d = st.local.duration();
      $('songsync').textContent = `${st.local.playing() ? 'Playing' : 'Paused'} local file · ${fmt(playbackTime())} / ${fmt(Number.isFinite(d) ? d : null)}`;
      renderSource();
      return;
    }
    $('songtitle').textContent = st.meta ? st.meta.title : 'No song';
    $('songartist').textContent = st.meta ? st.meta.artist : '';
    const pt = playbackTime();
    $('songsync').textContent = st.anchor ? `Manual sync · ${fmt(pt)}`
      : st.lines.length ? 'Click the line being sung to sync the lyrics' : 'Play a file here, or search for the lyrics';
    renderSource();
  }
  function renderSource() {
    $('songsrc').textContent = st.meta && st.meta.lyricsTitle ? `Lyrics: LRCLIB (${st.meta.lyricsTitle})` : '';
  }
  function setOffsetLabel() { $('offset').textContent = `${st.offset >= 0 ? '+' : ''}${st.offset.toFixed(1)} s`; }

  function bindControls() {
    $('offminus').onclick = () => { st.offset -= 0.1; setOffsetLabel(); render(); };
    $('offplus').onclick = () => { st.offset += 0.1; setOffsetLabel(); render(); };
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
      setOffsetLabel(); render(); renderHeader();
    });
    $('clearsheet').onclick = () => { st.chords = []; render(); };
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
        const res = await window.lrclib.search(q);
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
  // Common open-position voicings (string 6 -> 1, -1 = muted).
  const OPEN = {
    'C:maj': [-1, 3, 2, 0, 1, 0], 'A:maj': [-1, 0, 2, 2, 2, 0], 'G:maj': [3, 2, 0, 0, 0, 3], 'E:maj': [0, 2, 2, 1, 0, 0],
    'D:maj': [-1, -1, 0, 2, 3, 2], 'A:min': [-1, 0, 2, 2, 1, 0], 'E:min': [0, 2, 2, 0, 0, 0], 'D:min': [-1, -1, 0, 2, 3, 1],
    'C:7': [-1, 3, 2, 3, 1, 0], 'G:7': [3, 2, 0, 0, 0, 1], 'D:7': [-1, -1, 0, 2, 1, 2], 'E:7': [0, 2, 0, 1, 0, 0],
    'A:7': [-1, 0, 2, 0, 2, 0], 'C:maj7': [-1, 3, 2, 0, 0, 0], 'F:maj7': [-1, -1, 3, 2, 1, 0], 'A:maj7': [-1, 0, 2, 1, 2, 0],
    'D:min7': [-1, -1, 0, 2, 1, 1], 'A:min7': [-1, 0, 2, 0, 1, 0], 'E:min7': [0, 2, 0, 0, 0, 0],
    'D:sus4': [-1, -1, 0, 2, 3, 3], 'D:sus2': [-1, -1, 0, 2, 3, 0], 'A:sus4': [-1, 0, 2, 2, 3, 0], 'A:sus2': [-1, 0, 2, 2, 0, 0],
    'E:sus4': [0, 2, 2, 2, 0, 0],
  };
  // Movable shapes as fret offsets from the root fret (null = muted). E-shape: root on string 6;
  // A-shape: root on string 5. Offset 0 strings are barred.
  const E_SHAPE = {
    maj: [0, 2, 2, 1, 0, 0], min: [0, 2, 2, 0, 0, 0], 7: [0, 2, 0, 1, 0, 0], maj7: [0, 2, 1, 1, 0, 0],
    min7: [0, 2, 0, 0, 0, 0], sus4: [0, 2, 2, 2, 0, 0],
  };
  const A_SHAPE = {
    maj: [null, 0, 2, 2, 2, 0], min: [null, 0, 2, 2, 1, 0], 7: [null, 0, 2, 0, 2, 0], maj7: [null, 0, 2, 1, 2, 0],
    min7: [null, 0, 2, 0, 1, 0], sus2: [null, 0, 2, 2, 0, 0], sus4: [null, 0, 2, 2, 3, 0],
    dim: [null, 0, 1, 2, 1, null], aug: [null, 0, -1, -2, -2, null],
  };
  function shape(label) {
    if (OPEN[label]) return { frets: OPEN[label], barre: 0 };
    const [root, q] = label.split(':');
    const pc = PCS[root];
    const cands = [];
    if (E_SHAPE[q]) cands.push({ off: E_SHAPE[q], f: (pc - 4 + 12) % 12, from: 0 });
    if (A_SHAPE[q]) {
      let f = (pc - 9 + 12) % 12;
      if (q === 'aug' && f < 2) f += 12;  // the aug shape reaches two frets below the root
      cands.push({ off: A_SHAPE[q], f, from: 1 });
    }
    if (!cands.length) return null;
    const c = cands.reduce((a, b) => (b.f < a.f ? b : a));
    const frets = c.off.map((o) => (o === null ? -1 : c.f + o));
    const barred = c.f > 0 && c.off.filter((o) => o === 0).length >= 2 && !['dim', 'aug'].includes(q);
    return { frets, barre: barred ? c.f : 0, from: c.from };
  }
  window.chordShape = shape;  // exposed for the shape self-check
  window.chordDiagram = (label) => {
    if (!label || !label.includes(':')) return '';
    const sh = shape(label);
    if (!sh) return '';
    const { frets, barre, from } = sh;
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
  // Local file: "Artist - Title.mp3" -> lyrics lookup; the player drives the clock.
  function setLocal(src) {
    st.local = src;
    if (!src) { renderHeader(); return; }
    const base = src.name.replace(/\.[^.]+$/, '').replace(/_+/g, ' ').replace(/^\d+[\s.-]+/, '').trim();
    const [a, ...rest] = base.split(/\s+[-–—]\s+/);
    const meta = rest.length ? { artist: a.trim(), title: rest.join(' - ').trim() } : { artist: '', title: base };
    const go = () => {
      const d = src.duration();
      newSong(`local|${src.name}`, { ...meta, duration: Number.isFinite(d) ? d : null });
    };
    Number.isFinite(src.duration()) ? go() : setTimeout(go, 400);  // give the player a moment to read the duration
  }
  window.sheet = { addChord, setLocal, debug: () => st };
  bindControls();
  renderHeader(); render();
  setInterval(() => { highlight(); if (st.local || st.anchor) renderHeader(); }, 250);
})();
