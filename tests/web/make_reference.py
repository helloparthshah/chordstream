"""Reference output of the Python live engine for the browser-engine parity test (tests/web/parity.mjs).

usage: python tests/web/make_reference.py <audio> <bundle> <out_dir> [seconds]
Writes audio.f32 (16 kHz mono, raw little-endian float32) and ref.json (per-frame CQT, probs, events).
"""
import json
import sys
from pathlib import Path

from chordstream.data.dataset import load_audio
import numpy as np

from chordstream.stream import ChordStream

audio_path, bundle, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
secs = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
out.mkdir(parents=True, exist_ok=True)
y = load_audio(audio_path, 16000)[: int(secs * 16000)]
y = y.astype("<f4")
(out / "audio.f32").write_bytes(y.tobytes())
eng = ChordStream(bundle)
frames, probs = [], []
orig_push = eng.cqt.push


def tap(a):
    f = orig_push(a)
    frames.extend(f.tolist())
    return f


eng.cqt.push = tap
orig_run = eng.sess.run


def run(*a, **k):
    o = orig_run(*a, **k)
    probs.append(o[0][0].tolist())
    return o


eng.sess.run = run
events = []
for s in range(0, len(y), 1000):  # odd chunk size: exercises the pending-sample path
    events += [e.as_dict() for e in eng.push(y[s:s + 1000])]
keep = ("index", "name", "raw_index", "beat", "bpm", "beat_conf", "beat_name", "beat_t0", "key", "display_name")
ref = {"cqt": frames, "probs": probs, "events": [{k: e[k] for k in keep} for e in events]}
(out / "ref.json").write_text(json.dumps(ref), encoding="utf-8")
print(f"{len(events)} frames -> {out}")
