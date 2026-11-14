// Web Worker that runs the in-browser engine (engine.js) off the UI thread. Messages in:
//   {type: 'init', provider?: 'auto' | 'webgpu' | 'wasm'}, {type: 'audio', audio: Float32Array (16 kHz)},
//   {type: 'reset'}
// Messages out: {type: 'ready', provider, sample_rate, frame_seconds, bundle}, {type: 'error', message},
//   {type: 'events', events: [...chord events, same fields as the server's], ms: mean compute per frame}
import * as ort from '../ort/ort.webgpu.min.mjs';
import { Engine } from './engine.js';

ort.env.wasm.wasmPaths = new URL('../ort/', import.meta.url).href;
ort.env.wasm.numThreads = 1;  // threads need cross-origin isolation, which GitHub Pages can't enable
const MAX_BACKLOG = 25;       // ~1 s of audio waiting: drop the oldest instead of falling ever further behind

let eng = null, queue = Promise.resolve(), backlog = 0, dropped = 0;
onmessage = (m) => {
  const d = m.data;
  if (d.type === 'init') {
    Engine.load(ort, new URL('../model/', import.meta.url).href, { provider: d.provider })
      .then((e) => {
        eng = e;
        postMessage({ type: 'ready', provider: e.provider, timing: e.timing, sample_rate: e.sampleRate, frame_seconds: e.fs, bundle: e.meta.bundle });
      })
      .catch((err) => postMessage({ type: 'error', message: String(err && err.message || err) }));
  } else if (d.type === 'reset') {
    queue = queue.then(() => { eng && eng.reset(); backlog = 0; });
  } else if (d.type === 'audio' && eng) {
    if (backlog >= MAX_BACKLOG) { dropped++; return; }
    backlog++;
    queue = queue.then(async () => {
      const events = await eng.push(d.audio);
      backlog--;
      for (const e of events) delete e.probs;
      const ms = eng.computeMs, n = Math.min(ms.length, 25);
      let s = 0;
      for (let i = ms.length - n; i < ms.length; i++) s += ms[i];
      postMessage({ type: 'events', events, ms: n ? s / n : 0, dropped });
    }).catch((err) => postMessage({ type: 'error', message: String(err && err.message || err) }));
  }
};
