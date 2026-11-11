"""Causal key (key-signature) prior for chord smoothing.

Songs mostly stay in one key, and within a key a handful of chords dominate (I, IV, V, vi,
ii, iii and their minor-key counterparts). The network only sees ~5 s of context, so it
cannot know the song's key; this module estimates it from everything heard so far.

  * `fit_relative_profile` learns P(chord | key signature) from annotated training songs,
    key-relative and transposition-invariant (12 signatures: a major key and its relative
    minor share one diatonic set, so they need not be told apart).
  * `KeyPrior` keeps an exponentially decaying histogram of the smoothed chord posteriors,
    turns it into a posterior over the 12 signatures, and returns a chord prior that the
    HMM multiplies into its emissions with strength `gamma`.
"""

from __future__ import annotations

import numpy as np

from .vocab import IGNORE, N_CLASSES, NO_CHORD, transpose

_T = np.array([[transpose(c, k) for c in range(N_CLASSES)] for k in range(12)])  # _T[k, c] = c shifted up k


def _shift(h: np.ndarray, k: int) -> np.ndarray:
    """Histogram expressed relative to tonic k (i.e. transposed down by k)."""
    out = np.zeros_like(h)
    out[_T[(-k) % 12]] = h  # chord c -> c - k
    return out


def fit_relative_profile(label_seqs: list[np.ndarray], iters: int = 5) -> np.ndarray:
    """Key-relative chord distribution R (25,), estimated with a few rounds of hard EM."""
    hists = []
    for y in label_seqs:
        y = y[(y != IGNORE) & (y != NO_CHORD)]
        if len(y):
            h = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
            hists.append(h / h.sum())
    # Initialise with a plain diatonic template: I ii iii IV V vi (C major / A minor set).
    R = np.full(N_CLASSES, 0.01)
    for pc, minor in ((0, 0), (2, 1), (4, 1), (5, 0), (7, 0), (9, 1)):
        R[1 + pc + 12 * minor] = 1.0
    R /= R.sum()
    for _ in range(iters):
        acc = np.full(N_CLASSES, 1e-3)
        logR = np.log(R)
        for h in hists:
            k = int(np.argmax([_shift(h, k) @ logR for k in range(12)]))
            acc += _shift(h, k)
        acc[NO_CHORD] = 1e-3
        R = acc / acc.sum()
    return R


class KeyPrior:
    def __init__(self, R: np.ndarray, time_constant_s: float = 60.0, fps: float = 25.0, temperature: float = 1.0):
        self.logR = np.log(np.maximum(R, 1e-6))
        self.R = R
        self.decay = 1.0 if time_constant_s == np.inf else float(np.exp(-1.0 / (time_constant_s * fps)))
        self.temperature = temperature
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(N_CLASSES)
        self.mass = 0.0

    def key_posterior(self) -> np.ndarray:
        h = self.hist.copy()
        h[NO_CHORD] = 0.0
        if h.sum() <= 0:
            return np.full(12, 1 / 12)
        h /= h.sum()
        ll = np.array([_shift(h, k) @ self.logR for k in range(12)]) / self.temperature
        # Confidence grows with the amount of evidence (seconds of chords heard).
        ll *= min(1.0, self.mass / (25.0 * 20))
        p = np.exp(ll - ll.max())
        return p / p.sum()

    def prior(self) -> np.ndarray:
        pk = self.key_posterior()
        pri = np.zeros(N_CLASSES)
        for k in range(12):
            pri += pk[k] * _shift_up(self.R, k)
        pri[NO_CHORD] = 1.0 / N_CLASSES  # never bias for/against "no chord"
        return pri / pri.mean()

    def update(self, posterior: np.ndarray) -> None:
        self.hist = self.decay * self.hist + posterior
        self.mass = self.decay * self.mass + float(1.0 - posterior[NO_CHORD])


def _shift_up(R: np.ndarray, k: int) -> np.ndarray:
    """Key-relative distribution -> absolute distribution for tonic k."""
    out = np.zeros_like(R)
    out[_T[k]] = R  # relative chord c sits at absolute c + k
    return out
