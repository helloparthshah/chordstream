"""Tuning estimation: how far a recording's pitch sits from A440, in cents (-50..+50).

Spectral peaks (100-2000 Hz, 16384-point FFT, parabolic interpolation) are folded onto one semitone
and averaged on the circle, weighted by magnitude. Used to retune the CQT (fmin * 2^(cents/1200)) so
detuned recordings (tape speed, old film songs, instruments tuned to each other rather than A440)
land on the bins the model was trained on.
"""

from __future__ import annotations

import numpy as np


def estimate_tuning(audio: np.ndarray, sr: int = 16000, n_fft: int = 16384) -> float:
    win = np.hanning(n_fft); f = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (f > 100) & (f < 2000)
    z = 0j; wsum = 0.0
    for s in range(0, len(audio) - n_fft + 1, n_fft // 2):
        X = np.abs(np.fft.rfft(audio[s:s + n_fft] * win))
        if X.max() <= 0:
            continue
        pk = np.flatnonzero((X[1:-1] > X[:-2]) & (X[1:-1] > X[2:])) + 1
        pk = pk[band[pk]]
        if not len(pk):
            continue
        pk = pk[X[pk] > 0.05 * X[pk].max()]
        a, b, c = (np.log(X[pk + o] + 1e-12) for o in (-1, 0, 1))
        den = a - 2 * b + c
        d = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
        cents = 1200 * np.log2((pk + d) * sr / n_fft / 440.0)
        z += np.sum(X[pk] * np.exp(2j * np.pi * cents / 100)); wsum += X[pk].sum()
    return 0.0 if wsum == 0 else float(np.angle(z) / (2 * np.pi) * 100)
