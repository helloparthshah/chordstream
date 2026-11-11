# chordstream

Real-time, zero-lookahead chord recognition from live audio. Each 40 ms of audio
produces one decision (24 major/minor triads + N.C.), computed only from audio
already heard.

```
mic ─▶ causal CQT (40 ms hop) ─▶ causal CNN ─▶ banded causal Transformer (KV cache) ─▶ softmax ─▶ forward-HMM ─▶ chord
```

## Design notes

| Concern | Choice |
|---|---|
| Frontend | Causal variable-Q CQT, 16 kHz, 640-sample hop (40 ms), 24 bins/oct from C2 (6 octaves). Kernels end at "now"; low bins integrate the past ~0.2 s. |
| Causality | CNN pads time on the left only. Attention uses a causal band of 128 frames (~5 s) with ALiBi bias. Tests check that future input never changes past output. |
| Streaming cost | ONNX runs one step per frame: the CNN sees its 13-frame receptive field, and each layer keeps a 127-entry KV cache. A test checks this matches the full forward pass exactly. |
| Smoothing | HMM **forward filtering** only, never Viterbi backtracking, so smoothing adds no latency. Transitions = self-loop from median chord length plus a chord-change prior. |
| Vocabulary | Extended chords map to their triad (Cmaj7→C, Am9→Am). dim/aug/sus/power chords are `IGNORE`: left out of the loss, not forced into a wrong class. |
| UG "grammar" | Supplies (1) the HMM's chord-change matrix, counted in all 12 keys, and (2) progressions for the synthetic-audio renderer. |
| Pitch augmentation | Shift CQT bins by ±semitones and transpose the labels, applied to every training crop. |

## Workflow

```powershell
uv sync --extra train --extra mic            # CUDA torch on Windows/Linux

# 1. Grammar from saved Ultimate Guitar pages (parser only, no scraper; check UG's ToS)
uv run chordstream ug-parse path\to\saved_html --out data/ug.jsonl

# 2. Synthetic pre-training audio rendered from those progressions
uv run chordstream synth --progressions data/ug.jsonl --n 2000 --out data/synth

# 3. Real audio for the McGill Billboard annotations (CC0; extract salami + lab archives to data/billboard_raw)
uv sync --all-extras
uv run chordstream fetch-billboard --cookies-from-browser firefox --sleep 8   # resumable
uv run chordstream fetch-billboard --rescore --max-drift 0.3                 # re-tune thresholds offline
#    or pair annotations with audio you own:
uv run chordstream billboard-link --annotations billboard/McGill-Billboard --audio my_audio --out data/billboard

# 4. Cache features, pre-train on synth, fine-tune on real audio
uv run chordstream features data/synth --out data/cache_synth
uv run chordstream train --cache data/cache_synth --out models/pretrain --progressions data/ug.jsonl
uv run chordstream features data/billboard --out data/cache_bb
uv run chordstream train --cache data/cache_bb --init models/pretrain --out models/latest --lr 1e-4 --progressions data/ug.jsonl

# 5. Export + run
uv run chordstream export models/latest
uv run chordstream transcribe song.wav --ref song.lab --print   # simulated live stream + latency stats
uv run chordstream mic                                           # local input device
uv run chordstream serve                                         # http://localhost:8000 browser demo, ws://…/ws
```

Billboard ships annotations only. `fetch-billboard` searches YouTube and picks the result
whose length matches the annotation, preferring "- Topic" label uploads and skipping
live/cover/remix uploads. It then checks every download against its annotation:
- **Offset:** a chroma-vs-triad cross-correlation over ±10 s finds the time offset.
- **Wrong key:** the correct key must beat all 11 transpositions.
- **Drift:** the offsets found for the first and second halves must agree, which rules out
  different edits, re-pitched uploads and wrong songs.

Rejected audio goes to `_rejected/` and every decision is logged in `manifest.jsonl`.
YouTube's terms forbid downloading, so keep the audio local and don't redistribute it.
YouTube often bot-blocks anonymous downloads; `--cookies-from-browser` uses a signed-in
browser session. The fetcher stops after repeated bot or rate-limit errors instead of
retrying.

## Run locally or in Docker?

Train natively, since the GPU is used directly. Run live inference natively too: the
per-frame model is tiny, so CPU ONNX is as fast as a GPU would be, and a container on
Windows cannot open the microphone. The Docker image is a slim, PyTorch-free
deployment target for streaming over the LAN through the WebSocket API:

```powershell
docker build -t chordstream .
docker run -p 8000:8000 chordstream
```

## Live chord sheet

`chordstream serve`, then open http://localhost:8000.

- **Audio input:** use **Listen** with an input device (a mic, or a loopback bus such as a
  Voicemeeter "Out" bus or Stereo Mix), or **System audio** in Chrome/Edge. The level
  meter, spectrum and pitch-class bars (with the detected chord's tones lit) confirm that
  audio is arriving.
- **Now playing:** on Windows the server reads the OS media session (`/api/nowplaying`),
  which gives the title, artist and playback position from Spotify, YouTube Music, browsers
  and so on. It then fetches time-synced lyrics from [LRCLIB](https://lrclib.net)
  (`/api/lyrics`), cached in `data/lyrics_cache/`.
- **Placement:** each chord that holds for ≥ 320 ms is placed above the word being sung at
  that moment, Ultimate Guitar-style. Placement within a line is by timing, so it's
  approximate. Chords outside lyric lines go into Intro or Instrumental rows.
- **Sync controls:** **±** shifts the lyrics against the music, and clicking a lyric line
  as it is sung re-syncs everything. If the player doesn't report a position, clicking a
  line starts a manual clock.
- **Export:** **Copy** or **Download** gives a plain-text chords-over-lyrics sheet.
  `/#listen` auto-starts with the remembered input.

Requires `uv sync --extra desktop` (Windows media-session bindings).

**Word-level sync (optional):** `chordstream serve --asr small`, with `uv sync --extra asr`
and a GPU recommended.
- **Transcription:** every 2 s the server runs faster-whisper over the newest 8 s of audio,
  using the upcoming lyric lines as its prompt, and returns word timestamps (~0.5 s per
  window on an RTX 3050).
- **Alignment:** the page aligns those words to the lyric words with a fuzzy,
  time-constrained DP. Matched words are underlined and become timing anchors. Chords snap
  to the word being sung, and other words are interpolated between anchors.
- **Auto-sync:** line-start anchors correct the lyrics offset automatically. Turn it off
  with the **Auto-sync** toggle.
- **Script:** this works in any script Whisper writes in. Romanized lyrics of non-English
  songs match poorly, because Whisper writes Hindi in Devanagari.

## Results and experiments

Measured on 10 held-out Billboard songs (`chordstream.evaluate`), streaming-exact:

| Setup | Frame acc | Changes vs ref |
|---|---|---|
| Synthetic-only model | 46% | – |
| Fine-tuned on 76 real songs, raw | 74.3% | 4.2× |
| + causal HMM (uniform prior, default) | 73.9% | 1.9–2.0× |
| + beat-aware HMM (`beat.py`), even with *oracle* beats | 73.8% | 1.83× |
| + causal key prior (`key.py`) | ≤ 73.8% | 1.8–1.9× |
| **v2:** 207 real songs, transition-weighted loss (×3, 0.5 s), HMM power 1.5 | **75.5%** (77.0% on all 19 held-out) | 1.9× |

v2 also improved change timing: change-point F rose from 0.433 to 0.503 and median switching
lag fell from +120 to +80 ms on all held-out songs. More data alone (207 songs, plain loss)
reaches the same accuracy but keeps the +120 ms lag. Reproduce with
`chordstream train --transition-weight 3` and `chordstream evaluate models/<bundle>`.

Follow-up sweeps on 291 songs, compared on the same 25–37 held-out songs:
- **Transition weight:** 2 and 3 are within noise of each other (80.4% vs 80.1%), and 5 is
  slightly worse. Weight 3 gives the lowest lag (+40 ms).
- **Faster frontend** (`--gamma 10`, low-frequency windows 0.2 s → 0.08 s): 1 point *less*
  accurate (77.9% vs 78.8%), with lower change-F (0.499 vs 0.520) and no lag gain. Rejected,
  because blurred bass costs more than the shorter window saves.
- **Data scaling:** 207 → 394 songs (v3) is a tie with v2 (78.3% vs 78.4% on 37 held-out songs).
  v3 also scores only 79% on its *own training songs*, so the model under-fits rather than
  over-fits. More of the same data can't help; capacity, label noise and the causal limit
  are the candidates.

Error analysis:
- **Late switches:** 46% of errors are the previous chord still showing after a true change.
- **Near changes:** 43% of errors fall within 0.4 s of a change.
- **Lag:** median switching lag is +40 ms for the raw network and +120 ms after the HMM.

So the remaining accuracy is limited by how quickly the causal model commits to a new chord,
not by beat or key knowledge. `beat.py` and `key.py` are kept as evaluated experiments but
are not wired into the engine. The causal beat tracker currently scores beat F≈0.27 (tempo is
right on 6/10 songs, with half-tempo or off-beat locking on the rest). The next levers are
more real training data and transition-focused training.

## WebSocket protocol

Send binary messages of little-endian float32 mono PCM at 16 kHz, in any chunk size.
Each completed 40 ms frame returns one JSON event:
`{"type":"chord","time":12.36,"label":"A:min","name":"Am","confidence":0.94,"raw_index":22,"changed":false,...}`.
Send the text `reset` between songs. Set `CHORDSTREAM_ONLY_CHANGES=1` to receive only chord changes.

## Tests

`uv run pytest`: chord parsing, UG extraction, CQT pitch alignment, streaming/offline CQT
parity, CNN/Transformer causality, KV-cache step ≡ full forward, HMM anti-flicker.
