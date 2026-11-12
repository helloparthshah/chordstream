"""Feature-cache variants for experiments (E19 HPSS, E20 key input).

Both read an existing cache and write a new one with the same songs, labels and split, so the
only difference between runs is the feature change under test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..vocab import IGNORE, NO_CHORD

LOG_SCALE = 100.0  # features are log1p(LOG_SCALE * |CQT|)


def hpss_causal(feats: np.ndarray, k_time: int = 5, k_freq: int = 17, power: float = 2.0) -> np.ndarray:
    """Causal harmonic enhancement of log-CQT frames (soft-mask HPSS).

    Harmonic estimate: median over the current and previous k_time-1 frames (past only).
    Percussive estimate: median over k_freq neighbouring bins of the current frame.
    The frame is multiplied by the Wiener-like mask H^p / (H^p + P^p), which keeps sustained,
    pitched energy and suppresses broadband transients (drum hits). Uses no future frames.
    """
    lin = np.expm1(feats.astype(np.float64)) / LOG_SCALE
    T, F = lin.shape
    pad_t = np.concatenate([np.repeat(lin[:1], k_time - 1, 0), lin])
    win = np.lib.stride_tricks.sliding_window_view(pad_t, k_time, axis=0)  # (T, F, k_time)
    harm = np.median(win, axis=-1)
    half = k_freq // 2
    pad_f = np.pad(lin, ((0, 0), (half, half)), mode="edge")
    perc = np.median(np.lib.stride_tricks.sliding_window_view(pad_f, k_freq, axis=1), axis=-1)
    mask = harm**power / (harm**power + perc**power + 1e-12)
    return np.log1p(LOG_SCALE * lin * mask).astype(np.float32)


def key_signature_vector(labels: np.ndarray, R: np.ndarray) -> np.ndarray:
    """One-hot (12) key signature of a song from its labels (oracle: uses the annotations)."""
    from ..key import _shift

    y = labels[(labels != IGNORE) & (labels != NO_CHORD)]
    vec = np.zeros(12, np.float32)
    if len(y):
        h = np.bincount(y, minlength=len(R)).astype(np.float64)[: len(R)]
        h /= h.sum()
        logR = np.log(np.maximum(R, 1e-6))
        vec[int(np.argmax([_shift(h, k) @ logR for k in range(12)]))] = 1.0
    return vec


def build_variant(src: str | Path, out: str | Path, kind: str) -> int:
    from ..key import fit_relative_profile

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(src).glob("*.npz"))
    R = None
    if kind == "key":
        R = fit_relative_profile([np.load(f)["labels"].astype(np.int64) for f in files])
    for f in files:
        d = dict(np.load(f))
        feats = d["feats"].astype(np.float32)
        if kind == "hpss":
            d["feats"] = hpss_causal(feats).astype(np.float16)
        elif kind == "key":
            key = key_signature_vector(d["labels"].astype(np.int64), R)
            d["feats"] = np.concatenate([feats, np.tile(key, (len(feats), 1))], 1).astype(np.float16)
        else:
            raise ValueError(kind)
        np.savez_compressed(out / f.name, **d)
    return len(files)
