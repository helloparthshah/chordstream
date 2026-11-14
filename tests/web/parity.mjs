// Parity: the browser engine (src/chordstream/web/engine.js) vs the Python live engine on the same
// audio. Reference from tests/web/make_reference.py. usage: node parity.mjs <ref_dir> <web_bundle_dir>
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import * as ort from 'onnxruntime-web';
import { Engine } from '../../src/chordstream/web/engine.js';

const [refDir, bundleDir] = process.argv.slice(2);
globalThis.fetch = async (u) => {  // serve bundle files from disk
  const buf = readFileSync(new URL(u));
  return { ok: true, json: async () => JSON.parse(buf.toString()), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};
ort.env.wasm.numThreads = 1;
const ref = JSON.parse(readFileSync(path.join(refDir, 'ref.json')));
const raw = readFileSync(path.join(refDir, 'audio.f32'));
const audio = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const eng = await Engine.load(ort, pathToFileURL(path.resolve(bundleDir)).href + '/');

const events = [];
const t0 = performance.now();
for (let s = 0; s < audio.length; s += 1000) events.push(...await eng.push(audio.slice(s, s + 1000)));
const ms = (performance.now() - t0) / events.length;

let cqtErr = 0, probErr = 0;
const mism = {};
const FIELDS = ['index', 'name', 'raw_index', 'beat', 'bpm', 'beat_name', 'beat_t0', 'key', 'display_name'];
for (let i = 0; i < events.length; i++) {
  const e = events[i], r = ref.events[i];
  for (let j = 0; j < e.probs.length; j++) probErr = Math.max(probErr, Math.abs(e.probs[j] - ref.probs[i][j]));
  for (const f of FIELDS) if (e[f] !== r[f]) (mism[f] ??= []).push(i);
}
// CQT parity on a fresh front end
const { StreamingCQT } = await import('../../src/chordstream/web/engine.js');
const cqt = new StreamingCQT(eng.meta.frontend);
const fr = [];
for (let s = 0; s < audio.length; s += 1000) fr.push(...cqt.push(audio.slice(s, s + 1000)));
for (let i = 0; i < fr.length; i++) for (let k = 0; k < fr[i].length; k++) cqtErr = Math.max(cqtErr, Math.abs(fr[i][k] - ref.cqt[i][k]));

console.log(`frames ${events.length} / ref ${ref.events.length}   ${ms.toFixed(2)} ms per frame (node, wasm, 1 thread)`);
console.log(`max abs err: cqt ${cqtErr.toExponential(2)}  probs ${probErr.toExponential(2)}`);
for (const f of FIELDS) {
  const m = mism[f] || [];
  console.log(`${f.padEnd(13)} ${m.length ? `${m.length} mismatched frames, first ${m.slice(0, 5)}` : 'identical'}`);
}
const bad = events.length !== ref.events.length || ['index', 'display_name', 'beat_name'].some((f) => (mism[f] || []).length > events.length * 0.002);
process.exit(bad ? 1 : 0);
