"""Causal HMM smoothing of per-frame chord posteriors.

Uses the *forward* (filtering) recursion only, never Viterbi backtracking, so each
smoothed decision depends only on past and present frames: zero added latency.

Transition matrix A = p_stay * I + (1 - p_stay) * C, where C is a chord-change
distribution ("the grammar") learned from Ultimate Guitar progressions.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .vocab import IGNORE, N_CLASSES, NO_CHORD, transpose


def change_matrix_from_progressions(progressions: Iterable[list[int]], smoothing: float = 1.0,
                                    n_classes: int = N_CLASSES) -> np.ndarray:
    """Key-invariant chord-change probabilities (diagonal = 0), rows sum to 1.

    Every bigram is counted in all 12 transpositions, which is exactly the
    "transpose everything into all 12 keys" augmentation, done analytically.
    """
    counts = np.full((n_classes, n_classes), smoothing, dtype=np.float64)
    for prog in progressions:
        seq = [c for c in prog if c != IGNORE]
        for a, b in zip(seq, seq[1:]):
            if a == b:
                continue
            for k in range(12):
                counts[transpose(a, k), transpose(b, k)] += 1.0
    np.fill_diagonal(counts, 0.0)
    return counts / counts.sum(axis=1, keepdims=True)


def build_transition(change: np.ndarray | None, mean_chord_seconds: float, frame_seconds: float,
                     n_classes: int = N_CLASSES) -> np.ndarray:
    if change is None:
        change = np.ones((n_classes, n_classes))
        np.fill_diagonal(change, 0.0)
        change /= change.sum(axis=1, keepdims=True)
    p_stay = 1.0 - frame_seconds / mean_chord_seconds
    return p_stay * np.eye(len(change)) + (1.0 - p_stay) * change


class OnlineHMM:
    def __init__(self, A: np.ndarray, class_prior: np.ndarray | None = None, emission_power: float = 1.0):
        self.A = A.astype(np.float64)
        self.n = len(self.A)
        self.prior = np.ones(self.n) / self.n if class_prior is None else class_prior.astype(np.float64)
        self.power = emission_power
        # Decompose A = p_stay*I + (1-p_stay)*C so the change rate can be modulated per frame.
        self.p_stay = float(np.mean(np.diag(self.A)))
        C = self.A - self.p_stay * np.eye(self.n)
        np.fill_diagonal(C, 0.0)
        self.C = C / np.maximum(C.sum(1, keepdims=True), 1e-12)
        self.reset()

    def reset(self) -> None:
        self.alpha = np.zeros(self.n)
        self.alpha[NO_CHORD] = 1.0

    def step(self, probs: np.ndarray, change_scale: float = 1.0, chord_prior: np.ndarray | None = None) -> tuple[int, float]:
        """One causal update. change_scale multiplies the chord-change probability for this
        frame (1 = plain HMM). chord_prior (e.g. from KeyPrior, already raised to its strength)
        multiplies the emission likelihoods."""
        # Network posteriors -> scaled likelihoods (divide out the training class prior).
        lik = (np.clip(probs, 1e-8, 1.0) / self.prior) ** self.power
        if chord_prior is not None:
            lik = lik * chord_prior
        if change_scale == 1.0:
            pred = self.alpha @ self.A
        else:
            p_change = min(0.5, (1.0 - self.p_stay) * change_scale)
            pred = (1.0 - p_change) * self.alpha + p_change * (self.alpha @ self.C)
        a = pred * lik
        self.alpha = a / a.sum()
        s = int(self.alpha.argmax())
        return s, float(self.alpha[s])

    def filter(self, probs: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
        self.reset()
        if scales is None:
            return np.array([self.step(p)[0] for p in probs], dtype=np.int64)
        return np.array([self.step(p, float(c))[0] for p, c in zip(probs, scales)], dtype=np.int64)

    def save(self, path: str | Path) -> None:
        np.savez(path, A=self.A, prior=self.prior, power=self.power)

    @classmethod
    def load(cls, path: str | Path) -> "OnlineHMM":
        d = np.load(path)
        return cls(d["A"], d["prior"], float(d["power"]))
