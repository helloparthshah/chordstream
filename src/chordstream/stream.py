"""Real-time inference engine: numpy CQT + onnxruntime + causal HMM. No torch required.

Each 40 ms hop of audio yields exactly one decision, computed from audio up to and
including that hop: zero lookahead. A label-delay model (E23, `label_delay` = d frames) names the
chord of the frame d hops ago instead, so its decisions use d frames of "future" context; events
then carry `delay` = d x 40 ms. Chord-sheet commits are backdated to the frames they describe. Algorithmic latency is one hop (40 ms) plus
compute time (printed by `chordstream transcribe --bench`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .beat import BeatTracker
from .config import BundleConfig
from .cqt import StreamingCQT
from .hmm import OnlineHMM
from .spelling import KeySpeller
from .vocab import CLASS_NAMES


@dataclass
class ChordEvent:
    time: float  # seconds since stream start (end of the newest frame)
    index: int
    label: str  # Harte, e.g. "A:min"
    name: str  # display, e.g. "Am"
    confidence: float
    raw_index: int  # un-smoothed network argmax
    changed: bool
    beat: bool = False  # a beat is predicted to fall in this frame (display only)
    bpm: float = 0.0
    beat_conf: float = 0.0
    # Chord-sheet commit (F03): set when the live chord has held for `sheet_min_frames`; names that
    # chord and when it started (beat_t0, in described time). Used for the chord sheet /
    # progression, never for the live display. With v7 and 12 frames (480 ms): 83.4% majmin,
    # 1.03x the reference number of changes, vs 82.3% / beat-averaged posteriors (E16's method).
    # (Field names kept from the earlier per-beat version for client compatibility.)
    beat_label: str | None = None
    beat_name: str | None = None
    beat_t0: float | None = None
    delay: float = 0.0  # the chord fields describe the audio at time - delay (label-delay models)
    key: str | None = None  # estimated key signature (major tonic), used for spelling names

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class ChordStream:
    def __init__(self, bundle_dir: str | Path, threads: int = 2, smoothing: bool = True,
                 sheet_min_frames: int = 12):
        self.sheet_min_frames = sheet_min_frames
        import onnxruntime as ort

        bundle_dir = Path(bundle_dir)
        self.cfg = BundleConfig.load(bundle_dir / "config.json")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(bundle_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        hmm_path = bundle_dir / "hmm.npz"
        self.hmm = OnlineHMM.load(hmm_path) if smoothing and hmm_path.exists() else None
        st = np.load(bundle_dir / "init_state.npz")
        self.k0, self.v0 = st["k"].astype(np.float32), st["v"].astype(np.float32)
        self.rf = int(self.sess.get_inputs()[0].shape[1])
        self.reset()

    @property
    def sample_rate(self) -> int:
        return self.cfg.frontend.sample_rate

    def reset(self) -> None:
        self.cqt = StreamingCQT(self.cfg.frontend)
        self.beats = BeatTracker(fps=1.0 / self.cfg.frontend.frame_seconds)
        self.frames = np.zeros((1, self.rf, self.cfg.model.n_bins), np.float32)
        self.k, self.v = self.k0.copy(), self.v0.copy()
        self.n_frames = 0
        self.current = -1
        self.run_len = 0  # frames the live chord has held
        self.run_t0 = 0.0  # start of the current run, in described time
        self.committed = -1  # last chord written to the sheet
        self.speller = KeySpeller(fps=1.0 / self.cfg.frontend.frame_seconds)
        self.compute_ms: list[float] = []
        if self.hmm:
            self.hmm.reset()

    def push(self, audio: np.ndarray) -> list[ChordEvent]:
        """Feed mono float32 audio at `sample_rate`. Returns one event per completed 40 ms frame."""
        events = []
        t_cqt = time.perf_counter()
        new_frames = self.cqt.push(audio)
        cqt_ms = (time.perf_counter() - t_cqt) * 1000 / max(1, len(new_frames))
        for frame in new_frames:
            t0 = time.perf_counter()
            self.frames[0, :-1] = self.frames[0, 1:]
            self.frames[0, -1] = frame
            outs = self.sess.run(None, {"frames": self.frames, "k_cache": self.k, "v_cache": self.v})
            probs, self.k, self.v = outs[0][0], outs[1], outs[2]
            raw = int(probs.argmax())
            idx, conf = self.hmm.step(probs) if self.hmm else (raw, float(probs[raw]))
            b = self.beats.step(frame)
            self.compute_ms.append((time.perf_counter() - t0) * 1000 + cqt_ms)
            self.n_frames += 1
            changed = idx != self.current
            self.current = idx
            fs = self.cfg.frontend.frame_seconds
            now = self.n_frames * fs
            delay = self.cfg.model.label_delay * fs
            self.speller.update(idx)
            ev = ChordEvent(now, idx, CLASS_NAMES[idx], self.speller.name(idx), conf, raw, changed, b.beat,
                            round(b.bpm, 1), round(b.confidence, 2), delay=delay,
                            key=None if self.speller.key is None else self.speller.name_key())
            # Sheet: commit a chord once it has held sheet_min_frames, backdated to where it began,
            # so blips shorter than that never reach the sheet (the live readout still shows them).
            if changed:
                self.run_len, self.run_t0 = 0, now - delay - fs
            self.run_len += 1
            if self.run_len >= self.sheet_min_frames and idx != self.committed:
                self.committed = idx
                ev.beat_label, ev.beat_name, ev.beat_t0 = CLASS_NAMES[idx], self.speller.name(idx), round(self.run_t0, 3)
            events.append(ev)
        return events


def events_to_segments(events: list[ChordEvent], frame_seconds: float):
    segs = []
    for e in events:
        end = e.time - e.delay
        if segs and segs[-1][2] == e.label:
            segs[-1] = (segs[-1][0], end, e.label)
        else:
            segs.append((end - frame_seconds, end, e.label))
    return segs
