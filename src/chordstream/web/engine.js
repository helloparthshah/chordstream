// In-browser chordstream engine: a line-by-line port of the Python live path (stream.py), so the
// page can run with no server (GitHub Pages). Causal CQT (cqt.py) -> ONNX step model with KV caches
// (onnxruntime-web) -> causal HMM (hmm.py), plus the beat tracker (beat.py), key spelling
// (spelling.py), the chord-sheet commit and the Steady display. Events have the same fields as the
// server's WebSocket "chord" messages. tests/web/parity.mjs checks it against the Python engine.
//
// Usage: const eng = await Engine.load(ort, 'model/'); const events = await eng.push(float32At16k);

const QUALITIES = ['maj', 'min', '7', 'maj7', 'min7', 'dim', 'aug', 'sus2', 'sus4'];
const QUALITY_SUFFIX = ['', 'm', '7', 'maj7', 'm7', 'dim', 'aug', 'sus2', 'sus4'];
const QUALITY_INTERVALS = [[0, 4, 7], [0, 3, 7], [0, 4, 7, 10], [0, 4, 7, 11], [0, 3, 7, 10],
  [0, 3, 6], [0, 4, 8], [0, 2, 7], [0, 5, 7]];
const PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const NO_CHORD = 0;

export function decodeExt(idx) { return idx <= 0 ? null : [(idx - 1) % 12, Math.floor((idx - 1) / 12)]; }
export function harteName(idx) {
  const d = decodeExt(idx);
  return d ? `${PITCH_NAMES[d[0]]}:${QUALITIES[d[1]]}` : 'N';
}

// ------------------------------------------------------------------ causal CQT (cqt.py)
export class StreamingCQT {
  constructor(cfg) {
    this.cfg = cfg;
    const alpha = 2 ** (1 / cfg.bins_per_octave) - 1;
    this.kr = []; this.ki = []; this.len = [];
    for (let k = 0; k < cfg.n_bins; k++) {
      const f = cfg.fmin * 2 ** (k / cfg.bins_per_octave);
      const n = Math.ceil(cfg.filter_scale * cfg.sample_rate / (alpha * f + cfg.gamma));
      const win = new Float64Array(n);
      let wsum = 0;
      for (let i = 0; i < n; i++) { win[i] = n === 1 ? 1 : 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)); wsum += win[i]; }
      const re = new Float32Array(n), im = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const t = i - (n - 1);  // ends at 0 == "now"
        const ph = (2 * Math.PI * f * t) / cfg.sample_rate;
        re[i] = (2 * win[i] * Math.cos(ph)) / wsum;
        im[i] = (2 * win[i] * Math.sin(ph)) / wsum;
      }
      this.kr.push(re); this.ki.push(im); this.len.push(n);
    }
    this.L = Math.max(...this.len);
    this.buf = new Float32Array(this.L);
    this.pending = new Float32Array(0);
  }

  // Append audio; returns an array of Float32Array(n_bins) frames.
  push(audio) {
    const hop = this.cfg.hop_length, L = this.L, out = [];
    const p = new Float32Array(this.pending.length + audio.length);
    p.set(this.pending); p.set(audio, this.pending.length);
    let o = 0;
    while (p.length - o >= hop) {
      this.buf.copyWithin(0, hop);
      this.buf.set(p.subarray(o, o + hop), L - hop);
      o += hop;
      const fr = new Float32Array(this.cfg.n_bins);
      for (let k = 0; k < this.cfg.n_bins; k++) {
        const n = this.len[k], re = this.kr[k], im = this.ki[k], off = L - n;
        let a = 0, b = 0;
        for (let i = 0; i < n; i++) { const x = this.buf[off + i]; a += x * re[i]; b += x * im[i]; }
        fr[k] = Math.log1p(this.cfg.log_scale * Math.sqrt(a * a + b * b));
      }
      out.push(fr);
    }
    this.pending = p.slice(o);
    return out;
  }
}

// ------------------------------------------------------------------ causal HMM (hmm.py)
export class OnlineHMM {
  constructor(A, prior, power) {
    this.n = prior.length; this.A = A; this.prior = prior; this.power = power;
    this.alpha = new Float64Array(this.n); this.pred = new Float64Array(this.n);
    this.reset();
  }
  reset() { this.alpha.fill(0); this.alpha[NO_CHORD] = 1; }
  step(probs) {
    const n = this.n, A = this.A, al = this.alpha, pr = this.pred;
    pr.fill(0);
    for (let i = 0; i < n; i++) {
      const a = al[i];
      if (a === 0) continue;
      const row = i * n;
      for (let j = 0; j < n; j++) pr[j] += a * A[row + j];
    }
    let s = 0;
    for (let j = 0; j < n; j++) {
      const p = Math.min(1, Math.max(1e-8, probs[j]));
      const v = pr[j] * (p / this.prior[j]) ** this.power;
      al[j] = v; s += v;
    }
    let best = 0;
    for (let j = 0; j < n; j++) { al[j] /= s; if (al[j] > al[best]) best = j; }
    return [best, al[best]];
  }
}

// ------------------------------------------------------------------ beat tracker (beat.py)
const FPS = 25, HIST = 256;
const BPM_GRID = Array.from({ length: 290 }, (_, i) => 55 + 0.5 * i);

function interp(x, ys, left, right) {  // np.interp on xp = 0..len-1
  const n = ys.length;
  if (x < 0) return left === undefined ? ys[0] : left;
  if (x > n - 1) return right === undefined ? ys[n - 1] : right;
  const i = Math.floor(x);
  if (i >= n - 1) return ys[n - 1];
  const f = x - i;
  return ys[i] + f * (ys[i + 1] - ys[i]);
}

export class BeatTracker {
  constructor(fps = FPS, priorBpm = 110, priorOctaves = 0.9, tempoEvery = 10, phaseGain = 0.15) {
    this.fps = fps;
    this.lags = BPM_GRID.map((b) => (60 * fps) / b);
    this.prior = BPM_GRID.map((b) => Math.exp(-0.5 * (Math.log2(b / priorBpm) / priorOctaves) ** 2));
    this.tempoEvery = tempoEvery; this.gain = phaseGain;
    this.reset();
  }
  reset() {
    this.prev = null; this.onset = new Float32Array(HIST); this.fluxMean = 0; this.n = 0;
    this.period = (60 * this.fps) / 110; this.tempoConf = 0; this.phase = 0; this.conf = 0;
  }
  _onset(frame) {
    if (this.prev === null) { this.prev = frame; return 0; }
    let flux = 0;
    for (let i = 0; i < frame.length; i++) { const d = frame[i] - this.prev[i]; if (d > 0) flux += d; }
    flux = Math.fround(flux);
    this.prev = frame;
    this.fluxMean = 0.97 * this.fluxMean + 0.03 * flux;
    return Math.max(0, flux - this.fluxMean);
  }
  _updateTempo() {
    const m = Math.min(this.n, HIST);
    if (m < 4 * FPS) return;
    const o = this.onset.subarray(HIST - m);
    let mean = 0;
    for (let i = 0; i < m; i++) mean += o[i];
    mean /= m;
    const x = new Float64Array(m);
    let e = 0;
    for (let i = 0; i < m; i++) { x[i] = o[i] - mean; e += x[i] * x[i]; }
    if (e <= 1e-9) { this.tempoConf = 0; return; }
    const acf = new Float64Array(m);
    for (let l = 0; l < m; l++) { let s = 0; for (let i = 0; i + l < m; i++) s += x[i] * x[i + l]; acf[l] = s / e; }
    const at = (l) => interp(l, acf, undefined, 0);
    const score = this.lags.map((L, i) =>
      Math.max(0, at(L) + 0.5 * at(2 * L) + 0.25 * at(4 * L) + 0.25 * at(L / 2)) * this.prior[i]);
    let i = 0;
    for (let j = 1; j < score.length; j++) if (score[j] > score[i]) i = j;
    let cand = this.lags[i];
    let cur = 0;
    for (let j = 1; j < this.lags.length; j++) if (Math.abs(this.lags[j] - this.period) < Math.abs(this.lags[cur] - this.period)) cur = j;
    if (Math.abs(cand - this.period) > 0.5 && score[i] < 1.15 * score[cur]) { cand = this.lags[cur]; i = cur; }
    this.period = cand;
    this.tempoConf = Math.min(1, Math.max(0, at(cand)));
  }
  _measurePhase() {
    const P = this.period, o = this.onset, T = o.length - 1;
    const nPh = Math.ceil(P / 0.25), comb = new Float64Array(nPh);
    for (let a = 0; a < nPh; a++) {
      let s = 0;
      for (let k = 0; k < 6; k++) s += interp(T - a * 0.25 - k * P, o, 0) * 0.8 ** k;
      comb[a] = s;
    }
    let j = 0, mean = 0;
    for (let a = 0; a < nPh; a++) { mean += comb[a]; if (comb[a] > comb[j]) j = a; }
    mean /= nPh;
    let v = 0;
    for (let a = 0; a < nPh; a++) v += (comb[a] - mean) ** 2;
    const std = Math.sqrt(v / nPh);
    return [j * 0.25, (comb[j] - mean) / (std + 1e-6)];
  }
  step(frame) {
    this.onset.copyWithin(0, 1);
    this.onset[HIST - 1] = this._onset(frame);
    this.n += 1;
    if (this.n % this.tempoEvery === 0) this._updateTempo();
    const prevPhase = this.phase, P = this.period;
    const pred = (this.phase + 1) % P;
    let target;
    if (this.n >= 3 * FPS) {
      const [meas, peak] = this._measurePhase();
      const d = ((((meas - pred + P / 2) % P) + P) % P) - P / 2;  // circular difference
      this.phase = (((pred + this.gain * d) % P) + P) % P;
      const combConf = Math.min(1, Math.max(0, (peak - 1) / 2));
      target = combConf * Math.min(1, Math.max(0, this.tempoConf * 2.5));
    } else { this.phase = pred; target = 0; }
    this.conf = 0.95 * this.conf + 0.05 * target;
    return { beat: this.phase < prevPhase, bpm: this.period > 0 ? (60 * FPS) / this.period : 0, confidence: this.conf };
  }
}

// ------------------------------------------------------------------ key spelling (spelling.py)
const SHARPS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const FLATS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
const DEFAULT = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const MAJOR = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1];
const SHARP_KEYS = new Set([7, 2, 9, 4, 11, 6]), FLAT_KEYS = new Set([5, 10, 3, 8, 1]);
const spellRoot = (pc, key) => (SHARP_KEYS.has(key) ? SHARPS : FLAT_KEYS.has(key) ? FLATS : DEFAULT)[pc];

export class KeySpeller {
  constructor(fps = 25, timeConstant = 30, minSeconds = 8, hysteresis = 1.1) {
    this.decay = Math.exp(-1 / (fps * timeConstant));
    this.minFrames = Math.floor(minSeconds * fps);
    this.hyst = hysteresis;
    this.reset();
  }
  reset() { this.hist = new Float64Array(12); this.n = 0; this.key = null; }
  update(idx) {
    for (let i = 0; i < 12; i++) this.hist[i] *= this.decay;
    const d = decodeExt(idx);
    if (d) {
      for (const iv of QUALITY_INTERVALS[d[1]]) this.hist[(d[0] + iv) % 12] += 1;
      this.hist[d[0]] += 0.5;
      this.n += 1;
    }
    if (this.n < this.minFrames) return;
    const scores = [];
    for (let k = 0; k < 12; k++) { let s = 0; for (let i = 0; i < 12; i++) s += this.hist[i] * MAJOR[(i - k + 12) % 12]; scores.push(s); }
    let best = 0;
    for (let k = 1; k < 12; k++) if (scores[k] > scores[best]) best = k;
    if (this.key === null || scores[best] > this.hyst * scores[this.key]) this.key = best;
  }
  name(idx) { const d = decodeExt(idx); return d ? spellRoot(d[0], this.key) + QUALITY_SUFFIX[d[1]] : 'N.C.'; }
  nameKey() { const k = this.key; return `${spellRoot(k, k)} / ${spellRoot((k + 9) % 12, k)}m`; }
}

// ------------------------------------------------------------------ engine (stream.py)
export class Engine {
  static async load(ort, base = 'model/', opts = {}) {
    const get = async (f) => {
      const r = await fetch(base + f);
      if (!r.ok) throw new Error(`${base + f}: HTTP ${r.status}`);
      return r;
    };
    const meta = await (await get('meta.json')).json();
    const [model, state, hmm] = await Promise.all(['model.onnx', 'state.bin', 'hmm.bin']
      .map(async (f) => new Uint8Array(await (await get(f)).arrayBuffer())));
    // WebGPU first: the KV caches (k_out/v_out, ~10 MB) then stay on the GPU between steps and are
    // fed straight back in. WebAssembly (CPU) is the fallback when the browser has no WebGPU.
    const want = opts.provider || 'auto';
    let sess = null, provider = 'wasm';
    if (want !== 'wasm' && typeof navigator !== 'undefined' && navigator.gpu) {
      try {
        sess = await ort.InferenceSession.create(model, {
          executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
          preferredOutputLocation: { probs: 'cpu', k_out: 'gpu-buffer', v_out: 'gpu-buffer' },
        });
        provider = 'webgpu';
      } catch (err) {
        if (want === 'webgpu') throw err;
        console.warn('WebGPU unavailable, using WebAssembly', err);
      }
    }
    if (sess && want === 'auto') {
      // A GPU busy with other work (a game, training) can make WebGPU far slower than the CPU:
      // time a few warm-up steps and fall back to WebAssembly if they miss the budget.
      const ms = await Engine.timeSteps(ort, sess, meta, state, 8, 4 * (opts.gpuBudgetMs || 25));
      if (ms > (opts.gpuBudgetMs || 25)) {
        console.warn(`WebGPU too slow here (${ms.toFixed(0)} ms/step), using WebAssembly`);
        await sess.release?.();
        sess = null; provider = 'wasm';
      }
    }
    if (!sess) sess = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    const eng = new Engine(ort, sess, meta, state, hmm, opts);
    eng.provider = provider;
    return eng;
  }

  // Median wall time of n steps from the start state (the first, compile-heavy step excluded);
  // stops early, returning that step's time, as soon as one step exceeds giveUpMs.
  static async timeSteps(ort, sess, meta, stateBytes, n, giveUpMs = Infinity) {
    const all = new Float32Array(stateBytes.buffer, stateBytes.byteOffset, stateBytes.byteLength / 4);
    let k = new ort.Tensor('float32', all.slice(0, all.length / 2), meta.cache_shape);
    let v = new ort.Tensor('float32', all.slice(all.length / 2), meta.cache_shape);
    const frames = new ort.Tensor('float32', new Float32Array(meta.rf * meta.n_bins), [1, meta.rf, meta.n_bins]);
    const t = [];
    for (let i = 0; i <= n; i++) {
      const t0 = performance.now();
      const out = await sess.run({ frames, k_cache: k, v_cache: v });
      const dt = performance.now() - t0;
      if (k.location === 'gpu-buffer') { k.dispose(); v.dispose(); }
      k = out.k_out; v = out.v_out;
      if (!i) continue;
      t.push(dt);
      if (dt > giveUpMs) { t.length = 1; t[0] = dt; break; }
    }
    if (k.location === 'gpu-buffer') { k.dispose(); v.dispose(); }
    return t.sort((a, b) => a - b)[t.length >> 1];
  }

  constructor(ort, sess, meta, stateBytes, hmmBytes, { sheetMinFrames = 12, displayHold = 20, displayThr = 0.05 } = {}) {
    this.ort = ort; this.sess = sess; this.meta = meta;
    this.sheetMinFrames = sheetMinFrames; this.displayHold = displayHold; this.displayThr = displayThr;
    const all = new Float32Array(stateBytes.buffer, stateBytes.byteOffset, stateBytes.byteLength / 4);
    const half = all.length / 2;
    this.k0 = all.slice(0, half); this.v0 = all.slice(half);
    const h = new Float64Array(hmmBytes.buffer.slice(hmmBytes.byteOffset, hmmBytes.byteOffset + hmmBytes.byteLength));
    const n = meta.n_classes;
    this.hmm = new OnlineHMM(h.slice(0, n * n), h.slice(n * n, n * n + n), meta.hmm_power);
    this.fs = meta.frontend.hop_length / meta.frontend.sample_rate;
    this.sampleRate = meta.frontend.sample_rate;
    this.reset();
  }

  reset() {
    const m = this.meta;
    this.cqt = new StreamingCQT(m.frontend);
    this.beats = new BeatTracker(1 / this.fs);
    this.frames = new Float32Array(m.rf * m.n_bins);
    this.k?.dispose?.(); this.v?.dispose?.();  // free GPU buffers from the previous stream
    this.k = new this.ort.Tensor('float32', this.k0.slice(), m.cache_shape);
    this.v = new this.ort.Tensor('float32', this.v0.slice(), m.cache_shape);
    this.nFrames = 0; this.current = -1; this.runLen = 0; this.runT0 = 0; this.committed = -1; this.shown = -1;
    this.speller = new KeySpeller(1 / this.fs);
    this.hmm.reset();
    this.computeMs = [];
  }

  // Mono float32 audio at 16 kHz -> one event per completed 40 ms frame.
  async push(audio) {
    const m = this.meta, F = m.n_bins, events = [];
    const t0 = performance.now();
    const newFrames = this.cqt.push(audio);
    const cqtMs = (performance.now() - t0) / Math.max(1, newFrames.length);
    for (const frame of newFrames) {
      const t1 = performance.now();
      this.frames.copyWithin(0, F);
      this.frames.set(frame, (m.rf - 1) * F);
      const out = await this.sess.run({
        frames: new this.ort.Tensor('float32', this.frames.slice(), [1, m.rf, F]), k_cache: this.k, v_cache: this.v,
      });
      const probs = out.probs.data;
      if (this.k.location === 'gpu-buffer') { this.k.dispose(); this.v.dispose(); }
      this.k = out.k_out; this.v = out.v_out;
      let raw = 0;
      for (let j = 1; j < probs.length; j++) if (probs[j] > probs[raw]) raw = j;
      const [idx, conf] = this.hmm.step(probs);
      const b = this.beats.step(frame);
      this.computeMs.push(performance.now() - t1 + cqtMs);
      if (this.computeMs.length > 250) this.computeMs.shift();
      this.nFrames += 1;
      const changed = idx !== this.current;
      this.current = idx;
      const fs = this.fs, now = this.nFrames * fs, delay = m.label_delay * fs;
      this.speller.update(idx);
      const ev = {
        type: 'chord', time: now, index: idx, label: harteName(idx), name: this.speller.name(idx), confidence: conf,
        raw_index: raw, changed, beat: b.beat, bpm: Math.round(b.bpm * 10) / 10,
        beat_conf: Math.round(b.confidence * 100) / 100, beat_label: null, beat_name: null, beat_t0: null, delay,
        key: this.speller.key === null ? null : this.speller.nameKey(), display_label: '', display_name: '',
      };
      if (changed) { this.runLen = 0; this.runT0 = now - delay - fs; }
      this.runLen += 1;
      if (this.shown < 0 || (idx !== this.shown && (probs[this.shown] < this.displayThr || this.runLen >= this.displayHold))) this.shown = idx;
      ev.display_label = harteName(this.shown); ev.display_name = this.speller.name(this.shown);
      if (this.runLen >= this.sheetMinFrames && idx !== this.committed) {
        this.committed = idx;
        ev.beat_label = harteName(idx); ev.beat_name = this.speller.name(idx); ev.beat_t0 = Math.round(this.runT0 * 1000) / 1000;
      }
      ev.probs = probs;  // used by the parity test; the worker strips it
      events.push(ev);
    }
    return events;
  }
}
