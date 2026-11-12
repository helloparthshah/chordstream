# chordstream

Real-time chord recognition from live audio. Each 40 ms of audio produces one decision, computed
only from audio already heard. The live model (v8) names the chord of the audio 120 ms ago: that
short, trained-in delay lets each decision hear a little of what follows, which removes much of
the flicker at chord changes. It names 108 chords (12 roots × maj, min, 7, maj7, m7, dim, aug,
sus2, sus4) plus "no chord", and is trained on human-labelled data only: US Billboard hits plus
Bollywood songs whose chords come from Ultimate Guitar sheets. Chord names are spelled
for the song's estimated key (G♯m in E major, D♭ in A♭ major).

## Architecture

### System

Audio comes from a microphone, a loopback device, a shared tab or a local file. The browser streams
16 kHz PCM over a WebSocket, and the server answers each 40 ms frame with a chord event. The chord
sheet combines those events with the song's timed lyrics.

```mermaid
flowchart LR
    subgraph Browser["Browser (phone or PC)"]
        SRC["Mic / loopback / tab audio<br/>or Play file"] --> WL["AudioWorklet<br/>mono 16 kHz, 40 ms chunks"]
        UI["Live chord + diagram<br/>visualizer, BPM, timeline"]
        SHEET["Chord sheet<br/>chords over lyrics"]
    end
    subgraph Server["chordstream server (FastAPI)"]
        ENG["Streaming engine<br/>(see below)"]
        ASR["Whisper (optional)<br/>sung-word timestamps"]
        NP["Windows media session<br/>now playing + position"]
        LY["LRCLIB<br/>timed lyrics (cached)"]
    end
    WL -- "WebSocket: PCM" --> ENG
    WL -- "same audio" --> ASR
    ENG -- "chord per frame (120 ms behind),<br/>key, BPM, beat pulse" --> UI
    ENG -- "committed chords<br/>(held ≥ 480 ms, backdated)" --> SHEET
    ASR -- "word times" --> SHEET
    NP --> SHEET
    LY --> SHEET
```

### Streaming engine (one 40 ms step)

Everything is causal: each step uses only audio up to "now". The network is trained to name the
chord of 3 frames earlier (label delay, 120 ms), so its decision for a frame has heard 120 ms past it. The network runs incrementally, with
the CNN on its 13-frame receptive field and a per-layer KV cache, so a step costs about 2–3 ms of CPU.

```mermaid
flowchart LR
    PCM["640 new samples<br/>(40 ms)"] --> CQT["Causal variable-Q CQT<br/>144 bins, C2–B7<br/>kernels end at 'now'"]
    CQT --> NET["ChordNet step (ONNX)<br/>CNN + Transformer<br/>with KV cache"]
    NET --> P["Chord posteriors<br/>(109 classes)"]
    P --> HMM["Forward-only HMM<br/>no backtracking,<br/>no added latency"]
    HMM --> LIVE["Live chord<br/>(120 ms behind)"]
    HMM --> HOLD["Commit once held 480 ms,<br/>backdated to its start"]
    HOLD --> BEAT["Chord sheet + progression<br/>(blips never reach it)"]
    HMM --> KEY["Key estimate<br/>(~30 s chord histogram)"]
    KEY --> NAME["Spelled chord names<br/>G♯m vs A♭m"]
    CQT --> BT["Causal beat tracker<br/>onset flux, tempo, phase"]
    BT --> BPM["BPM + beat pulse (display only)"]
```

### Model (ChordNet)

```mermaid
flowchart TB
    IN["Log-CQT frames<br/>T × 144"] --> C1["Causal conv block × 3<br/>16 → 32 → 64 channels<br/>time padding on the left only,<br/>max-pool along frequency"]
    C1 --> PR["Linear projection<br/>→ 192-d per frame"]
    PR --> TR["Transformer encoder × 4<br/>4 heads, ALiBi distance bias,<br/>banded causal mask: last 128 frames (~5 s)"]
    TR --> H1["Chord head<br/>109 classes (exported)"]
    TR -. training only .-> H2["Root head (13)"]
    TR -. training only .-> H3["Bass head (13)"]
    TR -. training only .-> H4["Notes head<br/>12 × binary"]
```

The extra heads let rare chords and power chords (which have no class) still teach the shared layers.
Training targets are shifted 3 frames later (label delay), so output t names the chord at t − 3;
the network itself is unchanged. An offline teacher (the same network with a symmetric ±5 s band) is
kept only as a measuring stick: since v7 nothing is distilled from it.

### Training pipeline

```mermaid
flowchart TB
    UG["Ultimate Guitar progressions"] --> SYN["Synthetic renders<br/>(exact labels)"]
    SYN --> PRE["Pretrain on synthetic audio<br/>models/latest"]
    BB["McGill Billboard annotations"] --> FETCH["YouTube fetch<br/>duration match"]
    FETCH --> ALIGN{"Chroma alignment check<br/>offset, key, drift"}
    ALIGN -- "pass (394)" --> AUD["Audit + re-time labels<br/>exclude drifting songs (367 kept)"]
    AUD --> STU["Causal model<br/>human labels, 120 ms label delay<br/>+ transition weighting<br/>+ root/bass/notes heads"]
    PRE --> STU
    RARE["Rare-chord renders<br/>sus2/sus4/dim/aug (E27)"] -. in progress .-> STU
    YT["YouTube top Bollywood<br/>songs by year"] --> SHEETS["Ultimate Guitar chord sheets<br/>(matched + checked)"]
    SHEETS --> JA["Jump alignment<br/>chroma templates, no neural model"]
    JA --> CLEAN["Label cleaning<br/>bass evidence, capo key,<br/>drop least-confident 30%"]
    CLEAN --> STU
    STU --> EVAL["chordstream evaluate<br/>streaming-exact metrics"]
    EVAL --> EXP["ONNX export<br/>+ GitHub release"]
```

Every experiment behind these choices is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Design notes

| Concern | Choice |
|---|---|
| Frontend | Causal variable-Q CQT, 16 kHz, 640-sample hop (40 ms), 24 bins/oct from C2 (6 octaves). Kernels end at "now"; low bins integrate the past ~0.2 s. |
| Causality | CNN pads time on the left only. Attention uses a causal band of 128 frames (~5 s) with ALiBi bias. Tests check that future input never changes past output. |
| Streaming cost | ONNX runs one step per frame: the CNN sees its 13-frame receptive field, and each layer keeps a 127-entry KV cache. A test checks this matches the full forward pass exactly. |
| Smoothing | HMM **forward filtering** only, never Viterbi backtracking, so smoothing adds no latency. Transitions = self-loop (mean chord 5 s) plus a chord-change prior; emissions sharpened with power 2. |
| Look-ahead | **Label delay**: training targets are shifted 3 frames, so output t names the chord at t − 3 and every decision hears 120 ms past its frame. Worth +1.2–2.5 pts and fewer flips vs giving the same delay to the decoder (E23, E25). |
| Chord sheet | The sheet only takes a chord once the live chord has held 480 ms, then backdates it to where it began: about the reference number of chord changes (1.03×) and 83.4% majmin (F03). The live readout still shows every decision. |
| Spelling | A decaying (~30 s) histogram of the chords' notes picks the best-fitting key signature (with hysteresis); names use its sharps or flats. Display only. |
| Vocabulary | 109 classes: 12 roots × 9 qualities + N; the first 25 are the classic major/minor set. Richer chords fold into the nearest quality (Cadd9→C, C9→C7, hdim7→dim). Power chords have no class but still train the root/bass/notes heads. |
| UG "grammar" | Supplies (1) the HMM's chord-change matrix, counted in all 12 keys, and (2) progressions for the synthetic-audio renderer. |
| Pitch augmentation | Shift CQT bins by ±semitones and transpose the labels, applied to every training crop. |

## Pretrained models

Trained bundles are attached to GitHub Releases, so they don't bloat git history:

```powershell
gh release download model-v8 --pattern "*.zip" --dir models
Expand-Archive models/chordstream-v8.zip models; uv run chordstream serve --bundle models/v8
```

They were trained on copyrighted recordings, so keep them for personal use.

## Workflow

```powershell
uv sync --extra train --extra mic            # CUDA torch on Windows/Linux

# 1. Grammar from saved Ultimate Guitar pages
uv run chordstream ug-parse path\to\saved_html --out data/ug.jsonl

# 2. Synthetic pre-training audio rendered from those progressions
uv run chordstream synth --progressions data/ug.jsonl --n 2000 --out data/synth

# 3. Real audio for the McGill Billboard annotations (CC0; extract salami + lab archives to data/billboard_raw)
uv sync --all-extras
uv run chordstream fetch-billboard --cookies-from-browser firefox --sleep 8   # resumable
uv run chordstream fetch-billboard --rescore --max-drift 0.3                 # re-tune thresholds offline
#    or pair annotations with audio you own:
uv run chordstream billboard-link --annotations billboard/McGill-Billboard --audio my_audio --out data/billboard

# 4. Cache features and pre-train on synthetic audio (the init every model starts from)
uv run chordstream features data/synth --out data/cache_synth
uv run chordstream train --cache data/cache_synth --out models/latest --progressions data/ug.jsonl

# 5. Real-audio labels: cache, audit, re-time/clean (E14), extended-vocabulary twin (E18)
uv run chordstream features data/billboard --out data/cache_bb
uv run chordstream audit --cache data/cache_bb --out data/audit.csv
uv run chordstream clean                        # -> data/cache_bb_fix (367 songs)
uv run chordstream build-ext                    # -> data/cache_ext

# 6. Bollywood labels (E21; see below), then the mixed training folder
#    powershell -File experiments/2026-11-12_bollywood_clean.ps1 builds data/cache_ext_bolly2 and trains E21e (= v8)

# 7. Train the live recipe (v8 = E21e), evaluate, export
uv run chordstream train --init models/latest --cache data/cache_ext_bolly2 --vocab extended --label-delay 3 `
    --lr 3e-4 --epochs 80 --crops 2000 --transition-weight 3 --out models/v8
uv run chordstream evaluate models/v8 --cache data/cache_ext
uv run python experiments/analysis_bollywood.py models/v8   # Bollywood test songs
uv run chordstream export models/v8             # ONNX + parity check (HMM 5 s / power 2: see EXPERIMENTS E25)

# 8. Run (default bundle: models/v8)
uv run chordstream transcribe song.wav --ref song.lab --print   # simulated live stream + latency stats
uv run chordstream mic                                           # local input device
uv run chordstream serve                                         # http://localhost:8000 browser demo, ws://…/ws
```

**Bollywood data and rare chords** (see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) E21, E27):

```powershell
# E27: rare chord types from synthetic renders (training-only files, prefix synthx)
uv run chordstream synth --rare --n 400 --out data/synth_rare
uv run chordstream features data/synth_rare --extended synthx --out data/cache_rare

# E21: Bollywood songs with human chord labels from Ultimate Guitar sheets
uv run chordstream bollywood songs       # YouTube top-songs playlists, 2010-2024
uv run chordstream bollywood sheets      # match UG "Chords" sheets (requests spaced 1.5 s)
uv run chordstream bollywood validate    # drop look-alike / wrong-song matches
uv run chordstream bollywood audio       # download matched songs (Firefox session)
uv run chordstream bollywood cache       # CQT + sheet jump alignment -> data/cache_bolly
uv run chordstream bollywood clean       # cleaner labels: bass-aware alignment, capo key, low-confidence frames ignored -> data/cache_bolly2
```

The Bollywood stage fetches UG search and tab pages at a slow rate for personal research; check
Ultimate Guitar's terms before running it, and keep the sheets local.

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
- **Placement:** the engine commits a chord to the sheet once it has held 480 ms, backdated to
  where it began, and the page places it above the word being sung at that moment, Ultimate
  Guitar-style. Short blips show on the live readout but never reach the sheet. Placement within a line is by timing, so it's
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

## Gold labels for the Bollywood test songs

`https://<host>:8000/annotate` opens a correction page for the 61 Bollywood test songs (E21/E28). It
loads a song's audio and its sheet-aligned chords on a zoomable timeline; you play it (optionally
slower, looping one segment), click a segment and press **1–9** for the sheet's chords (**0** =
no chord, **X** = unsure, excluded from scoring), drag boundaries, **S**plit at the playhead or
**M**erge, and **P** plays the selected chord so it can be compared by ear. Changes auto-save to
`data/bollywood/gold/<id>.json`; **Mark song done** makes the Bollywood benchmark score that song
against the hand-corrected labels instead of the aligned ones.

## Results

The live model is **v8**: v7's recipe (extended vocabulary, 120 ms label delay, human labels only)
trained on Billboard plus 47 Bollywood songs with cleaned sheet-aligned labels. Models are chosen by
accuracy over **both** test sets.

| | v8 (live) | v7 | v6 (zero delay, distilled) |
|---|---|---|---|
| Bollywood, 23 test songs, majmin / full | **80.0% / 78.0%** | 68.9% / 61.3% | – |
| Bollywood, share of time on the sheet's chords | **92.8%** | 89.5% | – |
| Bollywood, 61 test songs (v3), majmin | **78.0%** | 60.7% | – |
| Billboard, 36 clean held-out songs, majmin / full | 81.8% / 70.5% | **83.3% / 73.0%** | 80.8% / 70.3% |
| **Overall** (mean of the two test sets, majmin) | **80.9%** | 76.1% | – |
| Chord sheet (480 ms hold, v7) | – | 83.4% majmin, 1.03× changes | – |

Bollywood accuracy is measured against sheet-aligned labels (about 82–84% accurate when checked on
Billboard); hand-corrected gold labels from the annotate page replace them song by song. Scaling to 152
Bollywood training songs (E28) did not beat v8. The
first fine-tuned model scored 73.8% on Billboard's original 10. A step costs about 2.4 ms of CPU per
40 ms frame. Every experiment, including the ones that didn't help, is in
**[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**, with commands and decisions. The batch scripts
are in [`experiments/`](experiments/).

## Using it from other devices on your LAN

Browsers only allow microphone access on
HTTPS pages or localhost, so serve HTTPS with a self-signed certificate:

```powershell
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout data/certs/key.pem -out data/certs/cert.pem `
  -subj "/CN=chordstream" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:<your LAN IP>"
uv run chordstream serve --host 0.0.0.0 --ssl-certfile data/certs/cert.pem --ssl-keyfile data/certs/key.pem
```

Open `https://<your LAN IP>:8000` and accept the certificate warning once per device. "Now
playing" and the lyrics follow the media playing on the server PC.

If Windows Firewall blocks Python, see [`tools/`](tools/): an admin script to allow it, or a Node
relay on a port that's already open.

## WebSocket protocol

Send binary messages of little-endian float32 mono PCM at 16 kHz, in any chunk size.
Each completed 40 ms frame returns one JSON event:
`{"type":"chord","time":12.36,"label":"A:min","name":"Am","confidence":0.94,"raw_index":22,"changed":false,"delay":0.12,"key":"C / Am",...}`.

| Field | Meaning |
|---|---|
| `time` | stream time at the end of the newest frame (s) |
| `label`, `name`, `index` | chord of the audio at `time − delay` (Harte label, key-spelled display name, class) |
| `delay` | the model's label delay in seconds (0.12 for v7) |
| `key` | estimated key signature, e.g. `"E / C#m"`, once known |
| `beat`, `bpm`, `beat_conf` | causal beat tracker (display only) |
| `beat_label`, `beat_name`, `beat_t0` | set when a chord is committed to the sheet: the chord and its start time (described time) |
Send the text `reset` between songs. Set `CHORDSTREAM_ONLY_CHANGES=1` to receive only chord changes.

## Tests

`uv run pytest`: chord parsing (Harte, free text, extended), UG extraction, CQT pitch alignment,
streaming/offline CQT parity, CNN/Transformer causality, KV-cache step ≡ full forward (also for the
optional sheet head), label-delay targets and realignment, HMM anti-flicker, key-aware spelling.
