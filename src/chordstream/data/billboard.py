"""McGill Billboard helpers.

The Billboard release ships annotations only. You must supply matching audio
yourself; it is not distributed. Two annotation forms are supported:

  * MIREX-style `majmin.lab` / `full.lab` (start  end  label). Preferred; used as-is.
  * `salami_chords.txt`. Converted to .lab here (bar-level timing, split evenly).

Training pairs are discovered as  <stem>.(wav|flac|mp3|ogg) + <stem>.lab  in a folder.
`link_dataset` builds such a folder from the Billboard id layout:
    annotations/<id>/majmin.lab   +   audio/<id>.<ext>
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
_TIME_RE = re.compile(r"^\s*([0-9.]+)\s+(.*)$")
_REPEAT_RE = re.compile(r"^x(\d+)$")


def salami_to_segments(text: str) -> list[tuple[float, float, str]]:
    lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = _TIME_RE.match(line)
        if m:
            lines.append((float(m.group(1)), m.group(2)))
    segs: list[tuple[float, float, str]] = []
    prev_chord = "N"
    for (t0, body), (t1, _) in zip(lines, lines[1:]):
        if "|" not in body:
            words = set(body.lower().replace(",", " ").split())
            if words & {"silence", "end", "z", "z'", "noise"}:
                segs.append((t0, t1, "N"))
            continue
        parts = body.split("|")
        bars = [p.split() for p in parts[1:-1]]
        tail = parts[-1].split()
        rep = next((int(m.group(1)) for tok in tail if (m := _REPEAT_RE.match(tok))), 1)
        bars = bars * rep
        if not bars:
            continue
        bar_dur = (t1 - t0) / len(bars)
        for b, toks in enumerate(bars):
            toks = [tk for tk in toks if not tk.startswith("(")] or ["."]
            beat = bar_dur / len(toks)
            for i, tk in enumerate(toks):
                chord = prev_chord if tk == "." else tk
                prev_chord = chord
                s = t0 + b * bar_dur + i * beat
                segs.append((s, s + beat, chord))
    return merge_segments(segs)


def merge_segments(segs):
    out = []
    for s, e, c in segs:
        if out and out[-1][2] == c and abs(out[-1][1] - s) < 1e-6:
            out[-1] = (out[-1][0], e, c)
        else:
            out.append((s, e, c))
    return out


def write_lab(segs, path: str | Path) -> None:
    Path(path).write_text("".join(f"{s:.6f}\t{e:.6f}\t{c}\n" for s, e, c in segs))


def read_lab(path: str | Path) -> list[tuple[float, float, str]]:
    segs = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 3:
            segs.append((float(p[0]), float(p[1]), p[2]))
    return segs


def find_pairs(folder: str | Path) -> list[tuple[Path, Path]]:
    pairs = []
    for lab in sorted(Path(folder).rglob("*.lab")):
        audio = next((lab.with_suffix(e) for e in AUDIO_EXTS if lab.with_suffix(e).exists()), None)
        if audio:
            pairs.append((audio, lab))
    return pairs


def link_dataset(annotations: str | Path, audio_dir: str | Path, out: str | Path) -> int:
    """Copy annotations/<id>/{majmin.lab|salami_chords.txt} next to audio/<id>.* into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audio = {p.stem: p for p in Path(audio_dir).iterdir() if p.suffix.lower() in AUDIO_EXTS}
    n = 0
    for d in sorted(Path(annotations).iterdir()):
        if not d.is_dir():
            continue
        key = d.name.lstrip("0") or "0"
        src = audio.get(d.name) or audio.get(key)
        if src is None:
            continue
        lab = d / "majmin.lab"
        if lab.exists():
            shutil.copy(lab, out / f"{d.name}.lab")
        elif (d / "salami_chords.txt").exists():
            write_lab(salami_to_segments((d / "salami_chords.txt").read_text()), out / f"{d.name}.lab")
        else:
            continue
        shutil.copy(src, out / f"{d.name}{src.suffix.lower()}")
        n += 1
    return n


def salami_beats(text: str) -> np.ndarray:
    """Reference beat times from salami_chords.txt bar lines (bars split evenly by the metre).

    Compound metres (6/8, 9/8, 12/8) count dotted-quarter beats. Used to evaluate the beat
    tracker; bar-level annotation means individual beats are approximate (+-1 frame).
    """
    num, den = 4, 4
    timed = []
    for line in text.splitlines():
        m = re.match(r"#\s*metre:\s*(\d+)/(\d+)", line)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            continue
        if line.startswith("#"):
            continue
        tm = _TIME_RE.match(line)
        if tm:
            per_bar = num // 3 if den == 8 and num % 3 == 0 else num
            timed.append((float(tm.group(1)), tm.group(2), per_bar))
    beats = []
    for (t0, body, per_bar), (t1, _, _) in zip(timed, timed[1:]):
        if "|" not in body:
            continue
        parts = body.split("|")
        bars = parts[1:-1]
        tail = parts[-1].split()
        rep = next((int(m.group(1)) for tok in tail if (m := _REPEAT_RE.match(tok))), 1)
        n_bars = len(bars) * rep
        if not n_bars:
            continue
        bar_dur = (t1 - t0) / n_bars
        for b in range(n_bars):
            for k in range(per_bar):
                beats.append(t0 + b * bar_dur + k * bar_dur / per_bar)
    return np.array(beats)
