"""Key-aware chord spelling (display only).

Chord classes have one canonical spelling (C#, Eb, F#, Ab, Bb). In E major G#m reads better than
"Abm"; in Ab major Db reads better than "C#". A decaying histogram of the chords heard (their
pitch classes, ~30 s memory) picks the best-fitting major-scale key signature, with hysteresis, and
roots are spelled with that signature's accidentals. Recognition is unaffected.
"""

from __future__ import annotations

import numpy as np

from .vocab import QUALITY_INTERVALS, QUALITY_SUFFIX, decode_ext

SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
DEFAULT = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MAJOR = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], float)
_SHARP_KEYS = {7, 2, 9, 4, 11, 6}  # G D A E B F#  (and their relative minors)
_FLAT_KEYS = {5, 10, 3, 8, 1}  # F Bb Eb Ab Db


def spell_root(pc: int, key: int | None) -> str:
    if key in _SHARP_KEYS:
        return SHARPS[pc]
    if key in _FLAT_KEYS:
        return FLATS[pc]
    return DEFAULT[pc]


def spelled_name(idx: int, key: int | None) -> str:
    d = decode_ext(idx)
    if d is None:
        return "N.C."
    return spell_root(d[0], key) + QUALITY_SUFFIX[d[1]]


class KeySpeller:
    def __init__(self, fps: float = 25.0, time_constant_s: float = 30.0, min_seconds: float = 8.0,
                 hysteresis: float = 1.1):
        self.decay = np.exp(-1.0 / (fps * time_constant_s))
        self.min_frames = int(min_seconds * fps)
        self.hyst = hysteresis
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(12)
        self.n = 0
        self.key: int | None = None  # major tonic of the signature (relative minor shares it)

    def update(self, idx: int) -> None:
        self.hist *= self.decay
        d = decode_ext(idx)
        if d is not None:
            root, q = d
            for iv in QUALITY_INTERVALS[q]:
                self.hist[(root + iv) % 12] += 1.0
            self.hist[root] += 0.5
            self.n += 1
        if self.n < self.min_frames:
            return
        scores = np.array([self.hist @ np.roll(_MAJOR, k) for k in range(12)])
        best = int(scores.argmax())
        if self.key is None or scores[best] > self.hyst * scores[self.key]:
            self.key = best

    def name(self, idx: int) -> str:
        return spelled_name(idx, self.key)

    def name_key(self) -> str:
        """e.g. "E / C#m" (the signature's major key and its relative minor)."""
        k = self.key
        return f"{spell_root(k, k)} / {spell_root((k + 9) % 12, k)}m"
