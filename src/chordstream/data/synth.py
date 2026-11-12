"""Render chord progressions to labelled audio.

This links the two halves of the data strategy: Ultimate Guitar supplies realistic
*sequences* (the grammar); this renderer turns them into audio with exact frame
labels, which adds aligned training data on top of the scarce real annotated audio.
Renders vary instrument timbre, voicing, inversion, rhythm, tempo, bass, drums,
added 7ths/9ths (still labelled with their triad), reverb-ish tails and noise.

Synthetic audio is used to pre-train. Fine-tune on real recordings (Billboard etc.).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..vocab import CLASS_NAMES, IGNORE, N_CLASSES, NO_CHORD, QUALITY_INTERVALS, decode, decode_ext, transpose
from .billboard import merge_segments, write_lab


def _midi_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


class Timbre:
    def __init__(self, rng: np.random.Generator):
        kind = rng.choice(["piano", "epiano", "pad", "pluck", "organ"])
        n = 12
        h = np.arange(1, n + 1)
        if kind == "piano":
            self.amps, self.decay, self.attack, self.inharm = 1 / h**1.1, rng.uniform(1.5, 4), 0.003, 3e-4
        elif kind == "epiano":
            self.amps, self.decay, self.attack, self.inharm = np.exp(-0.6 * (h - 1)), rng.uniform(1, 3), 0.004, 0
        elif kind == "pad":
            self.amps, self.decay, self.attack, self.inharm = 1 / h**1.6, 0.05, rng.uniform(0.05, 0.3), 0
        elif kind == "pluck":
            self.amps, self.decay, self.attack, self.inharm = 1 / h**0.9, rng.uniform(3, 7), 0.002, 1e-4
        else:  # organ: odd/even drawbar mix
            self.amps = np.where(h % 2 == 1, 1.0, 0.5) / h**0.7
            self.decay, self.attack, self.inharm = 0.0, 0.01, 0
        self.amps = self.amps * rng.uniform(0.6, 1.4, n)
        self.kind = kind

    def note(self, f0: float, dur: float, sr: int) -> np.ndarray:
        n = int(dur * sr)
        t = np.arange(n) / sr
        env = np.minimum(1.0, t / self.attack) * np.exp(-self.decay * t)
        rel = min(n, int(0.03 * sr))
        env[n - rel:] *= np.linspace(1, 0, rel)
        y = np.zeros(n)
        for k, a in enumerate(self.amps, start=1):
            fk = f0 * k * np.sqrt(1 + self.inharm * k * k)
            if fk >= sr / 2 - 200:
                break
            y += a * np.exp(-0.3 * (k - 1) * t * self.decay) * np.sin(2 * np.pi * fk * t + np.random.rand() * 6.28)
        return (env * y).astype(np.float32)


def _voicing(chord: int, rng) -> list[int]:
    if chord >= N_CLASSES:  # extended class (7, maj7, m7, dim, aug, sus2, sus4): exact intervals
        root, q = decode_ext(chord)
        notes = [48 + root + rng.integers(0, 2) * 12 + i for i in QUALITY_INTERVALS[q]]
        for _ in range(rng.integers(0, 3)):
            notes = notes[1:] + [notes[0] + 12]
        return notes
    d = decode(chord)
    if d is None:
        return []
    root, minor = d
    ivs = [0, 3 if minor else 4, 7]
    r = rng.random()
    if r < 0.25:
        ivs.append(10 if minor or rng.random() < 0.5 else 11)  # 7th
    elif r < 0.35:
        ivs.append(14)  # add9
    base = 48 + root + rng.integers(0, 2) * 12  # C3..B4 root
    notes = [base + i for i in ivs]
    for _ in range(rng.integers(0, 3)):  # random inversion
        notes = notes[1:] + [notes[0] + 12]
    if rng.random() < 0.4:
        notes.append(notes[0] + 12)
    return notes


def _drums(n: int, beat: float, sr: int, rng) -> np.ndarray:
    y = np.zeros(n, np.float32)
    step = int(beat * sr)
    kick = np.sin(2 * np.pi * 55 * np.arange(int(0.15 * sr)) / sr) * np.exp(-np.arange(int(0.15 * sr)) / (0.04 * sr))
    hat = rng.standard_normal(int(0.05 * sr)) * np.exp(-np.arange(int(0.05 * sr)) / (0.01 * sr))
    for i, s in enumerate(range(0, n - len(kick), step)):
        if i % 2 == 0:
            y[s : s + len(kick)] += 0.8 * kick
        else:
            y[s : s + len(hat)] += 0.4 * hat  # snare-ish noise on the backbeat
        hs = s + step // 2
        if hs + len(hat) < n:
            y[hs : hs + len(hat)] += 0.15 * hat
    return y


def render(progression: list[int], rng: np.random.Generator, sr: int = 16000, max_seconds: float = 60.0):
    """-> (audio float32, segments[(start, end, harte_label)])."""
    shift = int(rng.integers(0, 12))
    prog = [transpose(c, shift) for c in progression]
    bpm = rng.uniform(65, 150)
    beat = 60.0 / bpm
    timbre, bass_timbre = Timbre(rng), Timbre(rng)
    style = rng.choice(["block", "beats", "arp", "strum"])
    use_bass, use_drums = rng.random() < 0.7, rng.random() < 0.5

    lead_in = rng.uniform(0, 1.5)
    events, segs, t = [], [(0.0, lead_in, "N")], lead_in
    i = 0
    while t < max_seconds - 2 and prog:
        c = prog[i % len(prog)]
        i += 1
        beats = int(rng.choice([2, 4, 4, 4, 8]))
        dur = beats * beat
        label = "X" if c == IGNORE else CLASS_NAMES[c]
        if c == IGNORE:  # render an unlabelled sus4 so the model sees real out-of-vocab sound
            r = int(rng.integers(0, 12))
            notes, bass_note = [48 + r, 53 + r, 55 + r], 36 + r
        elif c == NO_CHORD:
            notes, bass_note = [], None
        else:
            notes = _voicing(c, rng)
            bass_note = 36 + (decode(c)[0] if c < N_CLASSES else decode_ext(c)[0])
        segs.append((t, t + dur, label))
        if notes:
            if style == "block":
                events += [(t, dur, m, 0.5) for m in notes]
            elif style == "beats":
                events += [(t + b * beat, beat, m, 0.4) for b in range(beats) for m in notes]
            elif style == "arp":
                sub = beat / 2
                for k in range(beats * 2):
                    events.append((t + k * sub, sub * 2, notes[k % len(notes)], 0.6))
            else:  # strum: staggered onsets
                for b in range(0, beats, 2):
                    events += [(t + b * beat + j * 0.015, 2 * beat, m, 0.45) for j, m in enumerate(notes)]
            if use_bass and bass_note is not None:
                events += [(t + b * beat, beat * 2, bass_note, 0.7) for b in range(0, beats, 2)]
        t += dur
    tail = rng.uniform(0.5, 2.0)
    segs.append((t, t + tail, "N"))
    n = int((t + tail) * sr)
    y = np.zeros(n, np.float32)
    for (s, d, m, a) in events:
        tim = bass_timbre if m < 45 else timbre
        x = tim.note(float(_midi_hz(m)), d + 0.15, sr) * a * rng.uniform(0.7, 1.0)
        i0 = int(s * sr)
        x = x[: max(0, n - i0)]
        y[i0 : i0 + len(x)] += x
    if use_drums:
        y[int(lead_in * sr):] += _drums(n - int(lead_in * sr), beat, sr, rng) * rng.uniform(0.2, 0.6)
    y += rng.standard_normal(n).astype(np.float32) * 10 ** rng.uniform(-4, -2.3)
    y *= rng.uniform(0.2, 0.9) / (np.abs(y).max() + 1e-6)
    return y, merge_segments(segs)


def render_dataset(progressions: list[list[int]], out: str | Path, n_songs: int, seed: int = 0, sr: int = 16000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in range(n_songs):
        prog = progressions[int(rng.integers(0, len(progressions)))]
        audio, segs = render(prog, rng, sr, max_seconds=float(rng.uniform(20, 60)))
        sf.write(out / f"synth_{k:05d}.wav", audio, sr, subtype="PCM_16")
        write_lab(segs, out / f"synth_{k:05d}.lab")


# A small built-in progression bank so the pipeline runs before any UG data exists.
_ROMAN = {
    "I": (0, False), "ii": (2, True), "iii": (4, True), "IV": (5, False), "V": (7, False),
    "vi": (9, True), "bVII": (10, False), "i": (0, True), "bIII": (3, False), "bVI": (8, False),
    "iv": (5, True), "v": (7, True),
}
_BANK = [
    "I V vi IV", "vi IV I V", "I vi IV V", "I IV V IV", "ii V I I", "I bVII IV I",
    "i bVI bIII bVII", "i iv v i", "i bVII bVI bVII", "I iii IV V", "IV I V vi", "I IV vi V",
]


def builtin_progressions() -> list[list[int]]:
    from ..vocab import class_index

    return [[class_index(*_ROMAN[r]) for r in p.split()] for p in _BANK]


# ---------------------------------------------------------------- rare chord types (E27)

def rare_progression(rng: np.random.Generator) -> list[int]:
    """A diatonic loop (in C; render() transposes) with idiomatic rare chords: sus2/sus4 that
    resolve, passing diminished chords, augmented links, and some 7th chords."""
    from ..vocab import QUALITIES, ext_index

    Q = {q: QUALITIES.index(q) for q in QUALITIES}
    deg = {"I": (0, "maj"), "ii": (2, "min"), "iii": (4, "min"), "IV": (5, "maj"), "V": (7, "maj"), "vi": (9, "min")}
    loop = [deg[d] for d in rng.choice(list(deg), size=int(rng.integers(3, 6)))]
    out = []
    for i, (root, q) in enumerate(loop):
        r = rng.random()
        nxt = loop[(i + 1) % len(loop)][0]
        if r < 0.2:  # suspension resolving to the chord
            out += [ext_index(root, Q[str(rng.choice(["sus4", "sus2"]))]), ext_index(root, Q[q])]
        elif r < 0.3:  # suspension standing alone
            out.append(ext_index(root, Q[str(rng.choice(["sus4", "sus2"]))]))
        elif r < 0.42:  # passing diminished a semitone below the next chord
            out += [ext_index(root, Q[q]), ext_index((nxt - 1) % 12, Q["dim"])]
        elif r < 0.56 and q == "maj":  # augmented link
            out += [ext_index(root, Q["maj"]), ext_index(root, Q["aug"])]
        elif r < 0.6:
            out.append(ext_index(root, Q[{"maj": str(rng.choice(["7", "maj7"])), "min": "min7"}[q]]))
        else:
            out.append(ext_index(root, Q[q]))
    return out


def render_rare_dataset(out: str | Path, n_songs: int, seed: int = 0, sr: int = 16000) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in range(n_songs):
        audio, segs = render(rare_progression(rng), rng, sr, max_seconds=float(rng.uniform(20, 60)))
        sf.write(out / f"rare_{k:05d}.wav", audio, sr, subtype="PCM_16")
        write_lab(segs, out / f"rare_{k:05d}.lab")
