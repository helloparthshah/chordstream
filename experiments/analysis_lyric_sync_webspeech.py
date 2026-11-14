"""W06 (continued): Chrome's built-in Web Speech API (cloud recognition) on the same lyric-sync task.
It only listens to live audio, so each 8 s window is played through Chrome's fake microphone in real
time (tests/... scratch runner webspeech_batch.mjs). No word timestamps: each final result's words are
spread over the time since the previous result, minus an assumed 0.8 s recognition delay.
Subset: 8 English + 8 Hindi songs, 4 windows each (the first 4 of each song's Whisper windows).
usage: python experiments/analysis_lyric_sync_webspeech.py <runner.mjs> <workdir>
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np
import soundfile as sf

from analysis_lyric_sync import SR, WIN, load_audio, load_songs, locate, lyric_tokens, norm, report, score, windows

runner, work = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
songs = load_songs()
pick = [s for s in songs if s["lang"] == "en"][:8] + [s for s in songs if s["lang"] == "hi"][:8]
jobs, truth = [], {}
for s in pick:
    y = load_audio(s["audio"], SR)
    for k, t0 in enumerate(windows(s, y)[:4]):
        wid = f"{s['id']}_{k}"
        win = np.concatenate([y[int(t0 * SR): int((t0 + WIN) * SR)], np.zeros(SR // 2)])
        wav = work / f"{wid}.wav"
        sf.write(str(wav), win.astype(np.float32), SR, subtype="PCM_16")
        jobs.append({"id": wid, "wav": str(wav.resolve()), "lang": "en-US" if s["lang"] == "en" else "hi-IN"})
        truth[wid] = (s, t0)
(work / "jobs.json").write_text(json.dumps(jobs))
if not (work / "results.json").exists():  # recognition runs in real time: reuse a finished run
    subprocess.run(["node", str(runner), str(work / "jobs.json"), str(work / "results.json")], check=True, cwd=runner.parent)
stats = {"en": [0, 0, 0, 0, 0.0], "hi": [0, 0, 0, 0, 0.0]}
heard = {"en": 0, "hi": 0}
for r in json.loads((work / "results.json").read_text(encoding="utf-8")):
    s, t0 = truth[r["id"]]
    _, toks = lyric_tokens(s["lrc"])
    words, prev = [], 0.0
    for e in r["results"]:
        if not e.get("final"):
            continue
        ws = e["text"].split()
        end = max(prev, e["t"] - 0.8)
        for i, w in enumerate(ws):
            words.append((prev + (end - prev) * i / max(1, len(ws)), norm(w)))
        prev = end
    heard[s["lang"]] += bool(words)
    score(stats[s["lang"]], locate(words, toks), t0)
report("webspeech", stats)
print("windows with any words:", heard)
