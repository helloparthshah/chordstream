"""Real-time inference engine: numpy CQT + onnxruntime + causal HMM. No torch required.

Each 40 ms hop of audio yields exactly one decision, computed from audio up to and
including that hop: zero lookahead. Algorithmic latency is one hop (40 ms) plus
compute time (printed by `chordstream transcribe --bench`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
        self.frames = np.zeros((1, self.rf, self.cfg.model.n_bins), np.float32)
        self.k, self.v = self.k0.copy(), self.v0.copy()
        self.n_frames = 0
        self.current = -1
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
            self.compute_ms.append((time.perf_counter() - t0) * 1000 + cqt_ms)
            self.n_frames += 1
            changed = idx != self.current
            self.current = idx
            events.append(ChordEvent(self.n_frames * self.cfg.frontend.frame_seconds, idx, CLASS_NAMES[idx],
                                     pretty_name(idx), conf, raw, changed))
        return events


def events_to_segments(events: list[ChordEvent], frame_seconds: float):
    segs = []
    for e in events:
        start = e.time - frame_seconds
        if segs and segs[-1][2] == e.label:
            segs[-1] = (segs[-1][0], e.time, e.label)
        else:
            segs.append((start, e.time, e.label))
    return segs
