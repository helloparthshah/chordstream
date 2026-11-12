"""chordstream command line.

Pipeline:
  ug-parse -> synth -> features -> train (pretrain) -> features(real) -> train --init (finetune)
  -> export -> transcribe / mic / serve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_ug_parse(a):
    from .data.ug import parse_directory, transposed

    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for rec in parse_directory(a.input, a.min_chords):
            for r in transposed(rec) if a.transpose_all else [rec]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} tabs -> {a.out}")


def cmd_billboard_link(a):
    from .data.billboard import link_dataset

    print(f"linked {link_dataset(a.annotations, a.audio, a.out)} songs into {a.out}")


def cmd_fetch_billboard(a):
    from .data.youtube import fetch_billboard, rescore

    if a.rescore:
        print(rescore(a.raw, a.out, a.min_score, a.min_margin, a.max_drift))
        return

    fetch_billboard(a.raw, a.out, a.limit, a.tolerance, a.min_score, a.min_margin, a.max_drift, a.sleep,
                    a.cookies_from_browser)


def cmd_synth(a):
    from .data.synth import builtin_progressions, render_dataset, render_rare_dataset
    from .data.ug import load_progressions

    if a.rare:
        render_rare_dataset(a.out, a.n, seed=a.seed)
        print(f"rendered {a.n} rare-chord songs -> {a.out}")
        return

    progs = load_progressions(a.progressions) if a.progressions else builtin_progressions()
    progs = [p for p in progs if p]
    render_dataset(progs, a.out, a.n, seed=a.seed)
    print(f"rendered {a.n} songs from {len(progs)} progressions -> {a.out}")


def cmd_features(a):
    from .config import FrontendConfig
    from .data.dataset import cache_features, cache_features_ext

    if a.extended:
        n = sum(cache_features_ext(src, a.out, a.extended, FrontendConfig(gamma=a.gamma), a.device) for src in a.src)
        print(f"cached {n} new songs (extended labels, prefix {a.extended}) -> {a.out}")
        return

    n = cache_features(a.src, a.out, FrontendConfig(gamma=a.gamma), a.device)
    print(f"cached {n} new songs -> {a.out}")


def cmd_bollywood(a):
    from .data import bollywood as b

    if a.stage == "songs":
        print(f"added {b.stage_songs(range(a.year_from, a.year_to + 1), a.per_year, a.playlists_per_year)} songs")
    elif a.stage == "sheets":
        print(b.stage_sheets())
    elif a.stage == "validate":
        print(b.stage_validate())
    elif a.stage == "audio":
        print(b.stage_audio(a.cookies_from_browser))
    elif a.stage == "cache":
        print(b.stage_cache(a.out))
    elif a.stage == "clean":
        print(b.stage_cache_clean())


def cmd_train(a):
    from .config import ModelConfig
    from .train import train

    from .vocab import N_CLASSES, N_EXT

    ext = a.vocab == "extended"
    mcfg = ModelConfig(context_frames=a.context, d_model=a.d_model, n_layers=a.layers,
                       n_classes=N_EXT if ext else N_CLASSES, structured=ext or a.structured, key_input=a.key_input,
                       label_delay=a.label_delay, sheet_delay=a.sheet_delay)
    train(a.cache, a.out, a.epochs, a.batch_size, a.lr, a.crops, a.init, a.progressions, a.holdout_prefix,
          a.workers, a.device, mcfg, a.transition_weight, a.transition_window, a.seed, FrontendConfig(gamma=a.gamma),
          a.eval_every, a.bidirectional, a.distill_weight, a.aux_weight, a.stability_weight, a.bass_dropout, a.random_eq)


def cmd_export(a):
    from .export import export_onnx

    export_onnx(a.bundle)


def cmd_transcribe(a):
    import time

    import numpy as np

    from .data.billboard import read_lab, write_lab
    from .data.dataset import frame_labels, load_audio
    from .stream import ChordStream, events_to_segments
    from .vocab import IGNORE

    eng = ChordStream(a.bundle, smoothing=not a.raw)
    y = load_audio(a.audio, eng.sample_rate)
    hop = eng.cfg.frontend.hop_length
    events, t0 = [], time.perf_counter()
    for s in range(0, len(y), hop):  # feed exactly like a live stream
        for e in eng.push(y[s : s + hop]):
            events.append(e)
            if a.print and e.changed:
                print(f"{e.time:8.2f}s  {e.name:6s} {e.confidence:.2f}")
    wall = time.perf_counter() - t0
    segs = events_to_segments(events, eng.cfg.frontend.frame_seconds)
    if a.out:
        write_lab(segs, a.out)
    ms = np.array(eng.compute_ms)
    dur = len(y) / eng.sample_rate
    print(f"{dur:.1f}s audio in {wall:.2f}s  (x{dur / wall:.1f} realtime)  "
          f"per-frame compute: mean {ms.mean():.2f} ms, p99 {np.percentile(ms, 99):.2f} ms, budget "
          f"{eng.cfg.frontend.frame_seconds * 1000:.0f} ms")
    if a.ref:
        ref = frame_labels(read_lab(a.ref), len(events), eng.cfg.frontend)
        pred = np.array([e.index for e in events])
        valid = ref != IGNORE
        print(f"frame accuracy vs reference: {(pred[valid] == ref[valid]).mean():.3f}")


def cmd_evaluate(a):
    from .evaluate import ORIGINAL_VAL, run

    for name, ids in (("all held-out", None), ("original 10", ORIGINAL_VAL)):
        for bundle in a.bundles:
            r = run(bundle, a.cache, ids=ids)
            for view in ("baseline", "majmin"):
                if view not in r:
                    continue
                b = r[view]
                tag = "" if view == "baseline" else " [majmin view]"
                print(f"{name:13s} {bundle:22s} songs {r['songs']:3d}  hmm {b['acc']:.3f}  "
                      f"changes {b['change_ratio']:.2f}x  change_f {b['change_f']:.3f}  lag {b['lag_ms']:+.0f} ms{tag}",
                      flush=True)


def cmd_audit(a):
    from .data.audit import run

    rows = run(a.cache, a.out, a.bundle)
    worst = sorted(rows, key=lambda r: -r["drift_s"])[:10]
    print(f"audited {len(rows)} songs -> {a.out}; largest drift:")
    for r in worst:
        print(f"  {r['drift_s']:.2f}s  {r['artist'][:24]} - {r['title'][:32]}")


def cmd_clean(a):
    from .data.audit import clean_cache

    print(clean_cache(a.cache, a.audit, a.out, a.max_drift))


def cmd_build_ext(a):
    from .data.dataset import build_extended_cache

    print(build_extended_cache(a.labs, a.cache_orig, a.cache_clean, a.out))


def cmd_pseudo_label(a):
    from .distill import pseudo_label

    print(pseudo_label(a.teacher, a.labelled, a.out, a.unlabelled_audio, a.unlabelled_cache))


def cmd_serve(a):
    import os

    import uvicorn

    os.environ["CHORDSTREAM_BUNDLE"] = a.bundle
    os.environ["CHORDSTREAM_ASR"] = a.asr
    # HTTPS lets other devices on the LAN use their microphone (browsers only allow
    # getUserMedia on secure origins or localhost).
    uvicorn.run("chordstream.server:app", host=a.host, port=a.port, log_level="info",
                ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile)


def cmd_mic(a):
    import queue

    import sounddevice as sd

    from .stream import ChordStream

    eng = ChordStream(a.bundle)
    q: queue.Queue = queue.Queue()
    hop = eng.cfg.frontend.hop_length
    if a.list_devices:
        print(sd.query_devices())
        return
    print("listening... Ctrl+C to stop")
    with sd.InputStream(samplerate=eng.sample_rate, channels=1, dtype="float32", blocksize=hop,
                        device=a.device, latency="low", callback=lambda d, *_: q.put(d[:, 0].copy())):
        try:
            while True:
                for e in eng.push(q.get()):
                    if e.changed:
                        print(f"\r{e.time:8.2f}s  {e.name:6s} ({e.confidence:.2f})        ", flush=True)
        except KeyboardInterrupt:
            pass


def cmd_listen(a):
    """Recognise whatever the system is playing (WASAPI loopback of the default speakers)."""
    import warnings

    import numpy as np
    import soundcard as sc
    import soxr

    from .stream import ChordStream

    warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)  # "data discontinuity"
    if a.list_devices:
        for spk in sc.all_speakers():
            print(spk.name)
        return
    spk = sc.get_speaker(a.speaker) if a.speaker else sc.default_speaker()
    loop = sc.get_microphone(id=str(spk.name), include_loopback=True)
    eng = ChordStream(a.bundle)
    sr_in = 48000
    rs = soxr.ResampleStream(sr_in, eng.sample_rate, 1, dtype="float32")
    block = sr_in * eng.cfg.frontend.hop_length // eng.sample_rate  # one 40 ms hop
    print(f"listening to system audio on '{spk.name}'... Ctrl+C to stop")
    last = None
    with loop.recorder(samplerate=sr_in, channels=2, blocksize=block) as rec:
        try:
            while True:
                x = rec.record(numframes=block).mean(axis=1).astype(np.float32)
                for e in eng.push(rs.resample_chunk(x)):
                    if e.name != last:
                        last = e.name
                        print(f"{e.time:8.2f}s  {e.name:6s} ({e.confidence:.2f})", flush=True)
        except KeyboardInterrupt:
            pass


from .config import FrontendConfig  # noqa: E402  (defaults for argparse)


def main(argv=None):
    p = argparse.ArgumentParser(prog="chordstream")
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("ug-parse", help="extract progressions from saved Ultimate Guitar HTML pages")
    s.add_argument("input")
    s.add_argument("--out", default="data/ug_progressions.jsonl")
    s.add_argument("--min-chords", type=int, default=4)
    s.add_argument("--transpose-all", action="store_true", help="write all 12 transpositions per tab")
    s.set_defaults(fn=cmd_ug_parse)

    s = sp.add_parser("billboard-link", help="pair Billboard annotations with your audio files")
    s.add_argument("--annotations", required=True)
    s.add_argument("--audio", required=True)
    s.add_argument("--out", default="data/billboard")
    s.set_defaults(fn=cmd_billboard_link)

    s = sp.add_parser("fetch-billboard", help="download + auto-align audio for Billboard annotations (yt-dlp)")
    s.add_argument("--raw", default="data/billboard_raw", help="extracted salami/ and lab/ archives")
    s.add_argument("--out", default="data/billboard")
    s.add_argument("--limit", type=int)
    s.add_argument("--tolerance", type=float, default=4.0, help="max duration mismatch (s)")
    s.add_argument("--min-score", type=float, default=0.2, help="mean chroma/template cosine")
    s.add_argument("--min-margin", type=float, default=0.08, help="score lead over best wrong key")
    s.add_argument("--max-drift", type=float, default=0.3, help="max first-half vs second-half offset (s)")
    s.add_argument("--rescore", action="store_true", help="re-apply thresholds to downloaded songs, no network")
    s.add_argument("--sleep", type=float, default=2.0, help="pause between songs (s)")
    s.add_argument("--cookies-from-browser", help="e.g. firefox: use that browser's YouTube session")
    s.set_defaults(fn=cmd_fetch_billboard)

    s = sp.add_parser("synth", help="render progressions into labelled training audio")
    s.add_argument("--progressions", help="jsonl from ug-parse (default: built-in bank)")
    s.add_argument("--out", default="data/synth")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rare", action="store_true",
                   help="E27: progressions with sus2/sus4/dim/aug/7ths (extended-vocabulary labels)")
    s.set_defaults(fn=cmd_synth)

    s = sp.add_parser("features", help="cache causal CQT features + frame labels")
    s.add_argument("src", nargs="+", help="folders of <stem>.wav + <stem>.lab")
    s.add_argument("--out", default="data/cache")
    s.add_argument("--device")
    s.add_argument("--gamma", type=float, default=FrontendConfig.gamma,
                   help="variable-Q offset (Hz): larger = shorter low-frequency windows, faster response")
    s.add_argument("--extended", metavar="PREFIX",
                   help="write extended-vocabulary labels (as data/cache_ext), files named PREFIX__<stem>; "
                        "prefix synthx = training only")
    s.set_defaults(fn=cmd_features)

    s = sp.add_parser("bollywood", help="E21: Bollywood songs + UG sheets -> aligned labels")
    s.add_argument("stage", choices=["songs", "sheets", "validate", "audio", "cache", "clean"])
    s.add_argument("--year-from", type=int, default=2010)
    s.add_argument("--year-to", type=int, default=2024)
    s.add_argument("--per-year", type=int, default=25)
    s.add_argument("--playlists-per-year", type=int, default=1)
    s.add_argument("--cookies-from-browser", default="firefox")
    s.add_argument("--out", default="data/cache_bolly")
    s.set_defaults(fn=cmd_bollywood)

    s = sp.add_parser("train")
    s.add_argument("--cache", default="data/cache")
    s.add_argument("--out", default="models/latest")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--crops", type=int, default=4000, help="random crops per epoch")
    s.add_argument("--context", type=int, default=128, help="attention context in frames (40 ms each)")
    s.add_argument("--d-model", type=int, default=192)
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--init", help="bundle dir to fine-tune from")
    s.add_argument("--progressions", help="UG jsonl used for the HMM transition prior")
    s.add_argument("--holdout-prefix", help="cache files with this prefix form the val set")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--device")
    s.add_argument("--transition-weight", type=float, default=1.0, help="loss weight just after chord changes")
    s.add_argument("--transition-window", type=float, default=0.5, help="seconds after a change to up-weight")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--eval-every", type=int, default=1, help="score held-out songs every N epochs (and the last)")
    s.add_argument("--bidirectional", action="store_true", help="train an offline teacher (sees future audio)")
    s.add_argument("--distill-weight", type=float, default=1.0, help="weight of the teacher-posterior loss")
    s.add_argument("--vocab", choices=["majmin", "extended"], default="majmin",
                   help="extended = 109 classes (maj min 7 maj7 m7 dim aug sus2 sus4) + structured heads")
    s.add_argument("--structured", action="store_true", help="add root/bass/notes heads (implied by extended)")
    s.add_argument("--aux-weight", type=float, default=0.5, help="weight of the structured-head losses")
    s.add_argument("--key-input", action="store_true", help="E20: cache has 12 key-signature columns after the CQT")
    s.add_argument("--stability-weight", type=float, default=0.0,
                   help="E22: penalise frame-to-frame prediction change where the chord is constant")
    s.add_argument("--bass-dropout", type=float, default=0.0,
                   help="E31: share of training crops with the low register (below a random 92-370 Hz cutoff) removed")
    s.add_argument("--random-eq", type=float, default=0.0,
                   help="E32: share of training crops passed through a random EQ curve (shelves, tilt, peaks/notches)")
    s.add_argument("--sheet-delay", type=int, default=0,
                   help="E26: add a chord-sheet head trained with this (longer) label delay")
    s.add_argument("--label-delay", type=int, default=0,
                   help="E23: output names the chord this many frames earlier (lookahead = delay x 40 ms)")
    s.add_argument("--gamma", type=float, default=FrontendConfig.gamma, help="frontend gamma (fresh models only; "
                   "must match the --gamma the cache was built with)")
    s.set_defaults(fn=cmd_train)

    s = sp.add_parser("evaluate", help="streaming-exact metrics on held-out annotated songs")
    s.add_argument("bundles", nargs="+")
    s.add_argument("--cache", default="data/cache_bb")
    s.set_defaults(fn=cmd_evaluate)

    s = sp.add_parser("audit", help="check annotated songs: alignment, drift, key, tuning (+ model agreement)")
    s.add_argument("--cache", default="data/cache_bb")
    s.add_argument("--out", default="data/audit.csv")
    s.add_argument("--bundle", help="optional model bundle for per-song agreement")
    s.set_defaults(fn=cmd_audit)

    s = sp.add_parser("clean", help="E14: re-time drifting labels, drop songs that still drift (-> cache_bb_fix)")
    s.add_argument("--cache", default="data/cache_bb")
    s.add_argument("--audit", default="data/audit.csv")
    s.add_argument("--out", default="data/cache_bb_fix")
    s.add_argument("--max-drift", type=float, default=1.0)
    s.set_defaults(fn=cmd_clean)

    s = sp.add_parser("build-ext", help="E18: extended-vocabulary twin of the cleaned cache (-> cache_ext)")
    s.add_argument("--labs", default="data/billboard")
    s.add_argument("--cache-orig", default="data/cache_bb")
    s.add_argument("--cache-clean", default="data/cache_bb_fix")
    s.add_argument("--out", default="data/cache_ext")
    s.set_defaults(fn=cmd_build_ext)

    s = sp.add_parser("pseudo-label", help="write teacher posteriors (soft targets) for distillation")
    s.add_argument("--teacher", required=True, help="bundle trained with --bidirectional")
    s.add_argument("--labelled", default="data/cache_bb_fix")
    s.add_argument("--out", default="data/cache_distill")
    s.add_argument("--unlabelled-audio", nargs="*", default=["data/billboard/_rejected"])
    s.add_argument("--unlabelled-cache", nargs="*", default=["data/cache_bb"],
                   help="caches whose songs not in --labelled are added without labels")
    s.set_defaults(fn=cmd_pseudo_label)

    s = sp.add_parser("export", help="export bundle to ONNX (+ parity check)")
    s.add_argument("bundle", nargs="?", default="models/latest")
    s.set_defaults(fn=cmd_export)

    s = sp.add_parser("transcribe", help="run the live engine over a file (simulated stream)")
    s.add_argument("audio")
    s.add_argument("--bundle", default="models/v9")
    s.add_argument("--out", help="write predicted .lab")
    s.add_argument("--ref", help="reference .lab for accuracy")
    s.add_argument("--raw", action="store_true", help="disable HMM smoothing")
    s.add_argument("--print", action="store_true", help="print chord changes as they happen")
    s.set_defaults(fn=cmd_transcribe)

    s = sp.add_parser("serve", help="FastAPI WebSocket server + browser mic demo")
    s.add_argument("--bundle", default="models/v9")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--ssl-certfile", help="serve HTTPS (needed for mic access from other devices)")
    s.add_argument("--ssl-keyfile")
    s.add_argument("--asr", default="off", help="Whisper size for lyric word alignment, e.g. small (needs the asr extra)")
    s.set_defaults(fn=cmd_serve)

    s = sp.add_parser("mic", help="live recognition from a local input device")
    s.add_argument("--bundle", default="models/v9")
    s.add_argument("--device", type=int)
    s.add_argument("--list-devices", action="store_true")
    s.set_defaults(fn=cmd_mic)

    s = sp.add_parser("listen", help="live recognition of audio the system is playing (Windows loopback)")
    s.add_argument("--bundle", default="models/v9")
    s.add_argument("--speaker", help="output device name substring (default: system default)")
    s.add_argument("--list-devices", action="store_true")
    s.set_defaults(fn=cmd_listen)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
