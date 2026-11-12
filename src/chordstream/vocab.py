"""Chord vocabularies.

majmin (25 classes, N_CLASSES): 24 major/minor triads + "N" (no chord / silence).
Index layout:  0 = N,  1..12 = C..B major,  13..24 = C..B minor.

extended (109 classes, N_EXT): 12 roots x QUALITIES + N, index 1 + 12*q + root. The first
25 indices are identical to majmin, so majmin models and labels are a strict subset.

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

QUALITIES = ["maj", "min", "7", "maj7", "min7", "dim", "aug", "sus2", "sus4"]
N_EXT = 1 + 12 * len(QUALITIES)
QUALITY_SUFFIX = ["", "m", "7", "maj7", "m7", "dim", "aug", "sus2", "sus4"]
QUALITY_INTERVALS = [(0, 4, 7), (0, 3, 7), (0, 4, 7, 10), (0, 4, 7, 11), (0, 3, 7, 10),
                     (0, 3, 6), (0, 4, 8), (0, 2, 7), (0, 5, 7)]
NO_ROOT = 12  # root/bass head class for "no chord"

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_NATURALS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ROOT_RE = re.compile(r"^([A-Ga-g])([#b♯♭]*)")


def class_index(root: int, minor: bool) -> int:
    return 1 + (root % 12) + (12 if minor else 0)


def ext_index(root: int, quality: int) -> int:
    return 1 + 12 * quality + (root % 12)


def decode_ext(idx: int) -> tuple[int, int] | None:
    """class index (either vocabulary) -> (root pitch class, quality index); None for N."""
    if idx <= 0:
        return None
    return (idx - 1) % 12, (idx - 1) // 12


def decode(idx: int) -> tuple[int, bool] | None:
    """class index -> (root pitch class, is_minor); None for N. Minor-third qualities count as minor."""
    d = decode_ext(idx)
    if d is None:
        return None
    return d[0], QUALITIES[d[1]] in ("min", "min7", "dim")


def harte_name(idx: int) -> str:
    d = decode_ext(idx)
    if d is None:
        return "N"
    return f"{PITCH_NAMES[d[0]]}:{QUALITIES[d[1]]}"


def pretty_name(idx: int) -> str:
    d = decode_ext(idx)
    if d is None:
        return "N.C."
    return PITCH_NAMES[d[0]] + QUALITY_SUFFIX[d[1]]


CLASS_NAMES = [harte_name(i) for i in range(N_EXT)]  # first 25 = majmin names


def transpose(idx: int, semitones: int) -> int:
    d = decode_ext(idx)
    if d is None:
        return idx  # N and IGNORE are pitch-invariant
    return ext_index(d[0] + semitones, d[1])


def to_majmin(idx: int) -> int:
    """Extended class -> majmin class (MIREX majmin reduction); IGNORE if no maj/min triad."""
    d = decode_ext(idx)
    if d is None:
        return idx
    q = QUALITIES[d[1]]
    if q in ("maj", "7", "maj7"):
        return class_index(d[0], False)
    if q in ("min", "min7"):
        return class_index(d[0], True)
    return IGNORE


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


def parse_freetext_ext(raw: str) -> int | None:
    """Like parse_freetext but into the extended vocabulary (maj, min, 7, maj7, m7, dim, aug, sus2,
    sus4). Extensions reduce to the nearest class (9 -> 7, m9 -> m7, add9/6 -> maj, m6 -> min,
    m7b5 -> dim); power chords ("5") return IGNORE; slash bass is dropped."""
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
    q = tail.split("/")[0].replace("(", "").replace(")", "")
    ql = q.lower()
    if q in ("5", "(5)"):
        return IGNORE
    if any(t in ql for t in ("dim", "°", "ø", "m7b5", "m7-5", "mb5", "m-5")):
        return ext_index(root, QUALITIES.index("dim"))
    if ql.startswith("aug") or q.startswith("+") or ("#5" in q and not q.startswith("m")):
        return ext_index(root, QUALITIES.index("aug"))
    if "sus2" in ql or q == "2":
        return ext_index(root, QUALITIES.index("sus2"))
    if "sus" in ql or q == "4":
        return ext_index(root, QUALITIES.index("sus4"))
    if q.startswith(("maj", "Maj", "MA", "M", "Δ")):
        return ext_index(root, QUALITIES.index("maj7" if any(d in q for d in ("7", "9", "11", "13")) else "maj"))
    if q.startswith(("min", "mi", "m", "-")):
        rest = q.lstrip("min-")
        if rest.startswith(("maj", "M")):
            return ext_index(root, QUALITIES.index("min"))
        return ext_index(root, QUALITIES.index("min7" if any(d in rest for d in ("7", "9", "11", "13")) else "min"))
    if q.startswith(("7", "9", "11", "13")):
        return ext_index(root, QUALITIES.index("7"))
    return ext_index(root, QUALITIES.index("maj"))  # "", 6, add9, ...


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


# ---------------------------------------------------------------- extended Harte parsing

_DEGREES = {"1": 0, "2": 2, "3": 4, "4": 5, "5": 7, "6": 9, "7": 11, "9": 14, "11": 17, "13": 21}
_SHORTHAND = {  # Harte shorthand -> (interval set, extended quality or None)
    "maj": ((0, 4, 7), "maj"), "min": ((0, 3, 7), "min"), "dim": ((0, 3, 6), "dim"), "aug": ((0, 4, 8), "aug"),
    "maj7": ((0, 4, 7, 11), "maj7"), "min7": ((0, 3, 7, 10), "min7"), "7": ((0, 4, 7, 10), "7"),
    "dim7": ((0, 3, 6, 9), "dim"), "hdim7": ((0, 3, 6, 10), "dim"), "minmaj7": ((0, 3, 7, 11), "min"),
    "maj6": ((0, 4, 7, 9), "maj"), "min6": ((0, 3, 7, 9), "min"), "6": ((0, 4, 7, 9), "maj"),
    "9": ((0, 4, 7, 10, 2), "7"), "maj9": ((0, 4, 7, 11, 2), "maj7"), "min9": ((0, 3, 7, 10, 2), "min7"),
    "11": ((0, 4, 7, 10, 2, 5), "7"), "maj11": ((0, 4, 7, 11, 2, 5), "maj7"), "min11": ((0, 3, 7, 10, 2, 5), "min7"),
    "13": ((0, 4, 7, 10, 2, 9), "7"), "maj13": ((0, 4, 7, 11, 2, 9), "maj7"), "min13": ((0, 3, 7, 10, 2, 9), "min7"),
    "sus2": ((0, 2, 7), "sus2"), "sus4": ((0, 5, 7), "sus4"), "5": ((0, 7), None), "1": ((0,), None),
}


def _degree(tok: str) -> int | None:
    m = re.fullmatch(r"([b#]*)(\d+)", tok.strip())
    if not m or m.group(2) not in _DEGREES:
        return None
    return (_DEGREES[m.group(2)] + m.group(1).count("#") - m.group(1).count("b")) % 12


def _quality_from_notes(notes: set[int]) -> str | None:
    """Best extended quality for an explicit interval set (e.g. "C:(1,b3,5)")."""
    for name, ivs in (("min7", {0, 3, 7, 10}), ("maj7", {0, 4, 7, 11}), ("7", {0, 4, 7, 10}),
                      ("min", {0, 3, 7}), ("maj", {0, 4, 7}), ("dim", {0, 3, 6}), ("aug", {0, 4, 8}),
                      ("sus4", {0, 5, 7}), ("sus2", {0, 2, 7})):
        if ivs <= notes:
            return name
    return None


def parse_harte_ext(label: str) -> tuple[int, int, int, int]:
    """Harte label -> (extended class or IGNORE, root head, bass head, 12-bit note mask).

    root/bass head: 0..11 pitch class, NO_ROOT for "N", IGNORE if unknown ("X").
    note mask: bit k set if pitch class k sounds; -1 if unknown. Chords outside the
    extended vocabulary (power chords, odd interval sets) get class IGNORE but keep their
    root, bass and notes, so the structured heads still learn from them.
    """
    s = label.strip()
    if s in ("N", "N.C.", ""):
        return NO_CHORD, NO_ROOT, NO_ROOT, 0
    if s == "X":
        return IGNORE, IGNORE, IGNORE, -1
    root_part, _, qual = s.partition(":")
    bass_tok = None
    if "/" in qual:
        qual, bass_tok = qual.split("/", 1)
    elif "/" in root_part:
        root_part, bass_tok = root_part.split("/", 1)
    parsed = _parse_root(root_part)
    if parsed is None or parsed[1]:
        return IGNORE, IGNORE, IGNORE, -1
    root = parsed[0]
    name, _, extra = (qual or "maj").partition("(")
    extra = extra.rstrip(")")
    if name:
        if name not in _SHORTHAND:
            return IGNORE, root, root, -1
        ivs, q = _SHORTHAND[name]
        notes = set(ivs)
    else:  # pure interval list
        notes, q = {0}, None
    for tok in filter(None, (t.strip() for t in extra.split(","))):
        if tok.startswith("*"):
            d = _degree(tok[1:])
            notes.discard(d)
        else:
            d = _degree(tok)
            if d is not None:
                notes.add(d)
    if not name:
        q = _quality_from_notes(notes)
    bass = root
    if bass_tok:
        d = _degree(bass_tok)
        bass = (root + d) % 12 if d is not None else root
    mask = 0
    for iv in notes:
        mask |= 1 << ((root + iv) % 12)
    cls = ext_index(root, QUALITIES.index(q)) if q else IGNORE
    return cls, root, bass, mask
