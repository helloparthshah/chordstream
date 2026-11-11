"""Chord vocabulary: 24 major/minor triads + "N" (no chord / silence).

Index layout:  0 = N,  1..12 = C..B major,  13..24 = C..B minor.

Chords that contain neither a major nor a minor triad (dim, hdim, aug, sus, power
chords, "X") map to IGNORE: they are excluded from the training loss rather than
being forced into the wrong class (same convention as MIREX "majmin" evaluation).
Extended chords collapse onto their triad: Cmaj7, C7, C9, C6 -> C;  Am7, Am9 -> Am.
"""

from __future__ import annotations

import re

N_CLASSES = 25
NO_CHORD = 0
IGNORE = -1

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_NATURALS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ROOT_RE = re.compile(r"^([A-Ga-g])([#b♯♭]*)")


def class_index(root: int, minor: bool) -> int:
    return 1 + (root % 12) + (12 if minor else 0)


def decode(idx: int) -> tuple[int, bool] | None:
    """class index -> (root pitch class, is_minor); None for N."""
    if idx <= 0:
        return None
    return (idx - 1) % 12, idx > 12


def harte_name(idx: int) -> str:
    d = decode(idx)
    if d is None:
        return "N"
    return f"{PITCH_NAMES[d[0]]}:{'min' if d[1] else 'maj'}"


def pretty_name(idx: int) -> str:
    d = decode(idx)
    if d is None:
        return "N.C."
    return PITCH_NAMES[d[0]] + ("m" if d[1] else "")


CLASS_NAMES = [harte_name(i) for i in range(N_CLASSES)]


def transpose(idx: int, semitones: int) -> int:
    d = decode(idx)
    if d is None:
        return idx  # N and IGNORE are pitch-invariant
    return class_index(d[0] + semitones, d[1])


def _parse_root(s: str) -> tuple[int, str] | None:
    m = _ROOT_RE.match(s)
    if not m:
        return None
    pc = _NATURALS[m.group(1).upper()]
    for acc in m.group(2):
        pc += 1 if acc in "#♯" else -1
    return pc % 12, s[m.end():]


# ---------------------------------------------------------------- free-text (UG) chords

_NO_CHORD_STRINGS = {"n.c.", "nc", "n.c", "n/c", "no chord", "(n.c.)", "n"}
_VALID_TAIL = re.compile(r"^[0-9a-zA-Z#b♯♭+\-°øΔ()/,.^]*$")


def parse_freetext(raw: str) -> int | None:
    """Parse a user-written chord symbol ("Am7", "Bbmaj7/D", "F#m7b5", "N.C.").

    Returns a class index, IGNORE for real-but-out-of-vocabulary chords,
    or None if the string does not look like a chord at all.
    """
    s = raw.strip().replace(" ", "")
    if not s:
        return None
    if s.lower() in _NO_CHORD_STRINGS:
        return NO_CHORD
    parsed = _parse_root(s)
    if parsed is None:
        return None
    root, tail = parsed
    if not _VALID_TAIL.match(tail):
        return None
    q = tail.split("/")[0]  # slash bass does not change the triad class
    ql = q.lower()

    if any(t in ql for t in ("dim", "°", "ø", "m7b5", "m7-5", "mb5", "m-5")):
        return IGNORE
    if ql.startswith("aug") or q.startswith("+") or "#5" in q and not q.startswith("m"):
        return IGNORE
    if "sus" in ql or q in ("5", "2", "4", "(5)"):
        return IGNORE
    if q.startswith(("maj", "Maj", "MA", "M", "Δ")):
        return class_index(root, minor=False)
    if q.startswith(("min", "mi", "m", "-")):
        return class_index(root, minor=True)
    return class_index(root, minor=False)  # "", 7, 9, 6, add9, 13, ...


# ---------------------------------------------------------------- Harte (.lab) chords

_MAJOR_Q = {"maj", "maj7", "7", "maj6", "9", "maj9", "11", "13", "maj11", "maj13", "6"}
_MINOR_Q = {"min", "min7", "minmaj7", "min6", "min9", "min11", "min13", "minmaj9"}


def parse_harte(label: str) -> int:
    """Parse Harte-syntax labels used by Billboard/Isophonics .lab files."""
    s = label.strip()
    if s in ("N", "N.C.", ""):
        return NO_CHORD
    if s == "X":
        return IGNORE
    root_part, _, qual = s.partition(":")
    root_part = root_part.split("/")[0]
    parsed = _parse_root(root_part)
    if parsed is None or parsed[1]:
        return IGNORE
    root = parsed[0]
    qual = qual.split("/")[0]
    if not qual:
        return class_index(root, minor=False)
    name, _, intervals = qual.partition("(")
    if not name:  # pure interval list, e.g. C:(1,b3,5)
        ivs = set(intervals.rstrip(")").split(","))
        if "b3" in ivs and "5" in ivs:
            return class_index(root, minor=True)
        if "3" in ivs and "5" in ivs:
            return class_index(root, minor=False)
        return IGNORE
    if name in _MAJOR_Q:
        return class_index(root, minor=False)
    if name in _MINOR_Q:
        return class_index(root, minor=True)
    return IGNORE  # dim, hdim7, aug, sus2, sus4, 1, 5, ...
