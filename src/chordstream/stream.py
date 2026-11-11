"""Real-time inference engine: numpy CQT + onnxruntime + causal HMM. No torch required.

Each 40 ms hop of audio yields exactly one decision, computed from audio up to and
including that hop: zero lookahead. A label-delay model (E23, `label_delay` = d frames) names the
chord of the frame d hops ago instead, so its decisions use d frames of "future" context; events
then carry `delay` = d x 40 ms and the per-beat chords are pooled over the frames they describe. Algorithmic latency is one hop (40 ms) plus
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
from .vocab import CLASS_NAMES, pretty_name


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
    # Set on the frame that closes a beat: the chord of the whole beat that just ended
    # (network posteriors averaged over the beat). One beat late by design; used for the
    # chord sheet / progression, never for the live display. Measured +1.3 pt and ~30%
    # less over-segmentation vs per-frame decisions on held-out Billboard songs.
    beat_label: str | None = None
    beat_name: str | None = None
    beat_t0: float | None = None
    delay: float = 0.0  # the chord fields describe the audio at time - delay (label-delay models)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class ChordStream:
    def __init__(self, bundle_dir: str | Path, threads: int = 2, smoothing: bool = True):
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
        self.beat_acc = np.zeros(self.cfg.model.n_classes, np.float64)
        self.beat_start = 0.0
        self.pending_beats: list[int] = []  # frame counts where a beat was flagged, not yet pooled
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
            probs, self.k, self.v = self.sess.run(
                None, {"frames": self.frames, "k_cache": self.k, "v_cache": self.v})
            probs = probs[0]
            raw = int(probs.argmax())
            idx, conf = self.hmm.step(probs) if self.hmm else (raw, float(probs[raw]))
            b = self.beats.step(frame)
            self.compute_ms.append((time.perf_counter() - t0) * 1000 + cqt_ms)
            self.n_frames += 1
            changed = idx != self.current
            self.current = idx
            fs = self.cfg.frontend.frame_seconds
            d = self.cfg.model.label_delay
            now = self.n_frames * fs
            ev = ChordEvent(now, idx, CLASS_NAMES[idx], pretty_name(idx), conf, raw, changed, b.beat,
                            round(b.bpm, 1), round(b.confidence, 2), delay=d * fs)
            if self.n_frames > d:  # probs describe frame n_frames - d; earlier ones describe nothing
                self.beat_acc += probs
            if b.beat:
                self.pending_beats.append(self.n_frames)
            # A beat closes once the frames it covers have all been described (d hops after the beat).
            if self.pending_beats and self.n_frames - d >= self.pending_beats[0]:
                end = self.pending_beats.pop(0) * fs
                j = int(self.beat_acc.argmax())
                ev.beat_label, ev.beat_name, ev.beat_t0 = CLASS_NAMES[j], pretty_name(j), round(self.beat_start, 3)
                self.beat_acc[:] = 0.0
                self.beat_start = end
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
