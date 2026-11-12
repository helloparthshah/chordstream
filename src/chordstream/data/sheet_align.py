"""Timed chord labels from an untimed chord sheet (E21), without any neural model.

A chord sheet (e.g. Ultimate Guitar) gives the chords a song uses and which follows which, but no
times. Here the sheet becomes a song-specific HMM grammar: states are the sheet's chords (+ "no
chord"), transitions only along the sheet's chord-to-chord bigrams (plus a small escape to any sheet
chord, since sheets omit repeats and fills). Emissions are cosine similarities between a smoothed
CQT chroma and binary chord templates. Offline Viterbi gives the timing; all 12 transpositions of
the sheet are tried (capo / different key) and the best-fitting one is kept with its margin.

The chord identities are human; only the timing is estimated, by signal processing.
"""

from __future__ import annotations

import numpy as np

from ..vocab import IGNORE, NO_CHORD, QUALITY_INTERVALS, decode_ext, transpose
from .youtube import chroma_from_cqt


def template(c: int) -> np.ndarray:
    t = np.zeros(12, np.float32)
    d = decode_ext(c)
    if d is None:  # no chord: flat (after mean removal: zero vector)
        return t
    root, q = d
    for iv in QUALITY_INTERVALS[q]:
        t[(root + iv) % 12] = 1.0
    t[root] += 0.5  # the root is usually the strongest pitch class
    t -= t.mean()
    return t / np.linalg.norm(t)


def smooth_chroma(feats: np.ndarray, bins_per_semitone: int, width: int = 9) -> np.ndarray:
    ch = chroma_from_cqt(feats, bins_per_semitone)
    k = np.ones(width, np.float32) / width
    ch = np.stack([np.convolve(ch[:, i], k, mode="same") for i in range(12)], 1)
    return ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-6)


def sheet_grammar(seq: list[int], escape: float = 0.05) -> tuple[list[int], np.ndarray]:
    """States (sheet chords + N) and a change matrix (rows sum to 1, diagonal 0)."""
    states = [NO_CHORD] + sorted({c for c in seq if c not in (NO_CHORD, IGNORE)})
    ix = {c: i for i, c in enumerate(states)}
    K = len(states)
    counts = np.zeros((K, K))
    s = [c for c in seq if c != IGNORE]
    for a, b in zip(s, s[1:]):
        if a != b:
            counts[ix[a], ix[b]] += 1
    counts[0, 1:] += 1  # songs start/stop with silence; N can go anywhere and come from anywhere
    counts[1:, 0] += 0.2
    C = counts / np.maximum(counts.sum(1, keepdims=True), 1e-9)
    C = (1 - escape) * C + escape * (1 - np.eye(K)) / (K - 1)
    np.fill_diagonal(C, 0)
    return states, C / C.sum(1, keepdims=True)


def viterbi(logE: np.ndarray, logA: np.ndarray) -> tuple[np.ndarray, float]:
    T, K = logE.shape
    delta = logE[0] + np.log(np.r_[1.0, np.full(K - 1, 1e-3)])
    back = np.zeros((T, K), np.int32)
    for t in range(1, T):
        m = delta[:, None] + logA
        back[t] = m.argmax(0)
        delta = m.max(0) + logE[t]
    path = np.zeros(T, np.int64)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(delta.max())


def align_sheet(feats: np.ndarray, seq: list[int], bins_per_semitone: int = 2, frame_seconds: float = 0.04,
                mean_chord_seconds: float = 2.0, beta: float = 8.0, shifts=range(12)) -> dict:
    """Return frame labels for the best transposition of the sheet, plus fit diagnostics."""
    ch = smooth_chroma(feats, bins_per_semitone)
    energy = feats.mean(1)
    quiet = energy < np.percentile(energy, 5) * 1.05
    p_stay = 1 - frame_seconds / mean_chord_seconds
    results = []
    for k in shifts:
        states, C = sheet_grammar([transpose(c, k) if c not in (NO_CHORD, IGNORE) else c for c in seq])
        A = p_stay * np.eye(len(states)) + (1 - p_stay) * C
        tpl = np.stack([template(c) for c in states])
        cos = ch @ tpl.T  # (T, K); N's template is 0 -> cos 0
        cos[:, 0] = np.where(quiet, 0.6, 0.15)  # "no chord": favoured only in near-silence
        path, ll = viterbi(beta * cos, np.log(A + 1e-12))
        fit = float(cos[np.arange(len(path)), path][path != 0].mean()) if (path != 0).any() else 0.0
        results.append({"shift": k, "labels": np.array(states)[path], "fit": fit, "ll": ll})
    results.sort(key=lambda r: -r["ll"])
    best = results[0]
    best["margin"] = (best["ll"] - results[1]["ll"]) / len(feats) if len(results) > 1 else None
    return best


def ordered_viterbi(cos: np.ndarray, seq: list[int], states: list[int], p_stay: float, beta: float,
                    skip: float = 0.02, jumps: list[int] | None = None, jump: float = 1e-4) -> tuple[np.ndarray, float]:
    """Left-to-right forced alignment of the sheet's chord *sequence* (McVicar et al.'s jump
    alignment when `jumps` lists positions that any line end may jump back/forward to)."""
    ix = {c: i for i, c in enumerate(states)}
    L = len(seq)
    E = beta * cos[:, [ix[c] for c in seq]]  # (T, L)
    T = len(E)
    ls, la, lk, lj = np.log(p_stay), np.log((1 - p_stay) * (1 - skip)), np.log((1 - p_stay) * skip), np.log(jump)
    delta = np.full(L, -np.inf)
    delta[0] = E[0, 0]
    if L > 1:
        delta[1] = E[0, 1] + np.log(1e-3)
    back = np.zeros((T, L), np.int32)
    idx = np.arange(L)
    jmp = np.array(jumps if jumps else [], int)
    for t in range(1, T):
        cand = np.stack([delta + ls, np.r_[-np.inf, delta[:-1]] + la, np.r_[-np.inf, -np.inf, delta[:-2]] + lk])
        src = np.stack([idx, idx - 1, idx - 2])
        best = cand.argmax(0)
        val = cand[best, idx]
        b = src[best, idx]
        if len(jmp):  # jump from the best state anywhere to a section/line start
            j0 = int(delta.argmax())
            v = delta[j0] + lj
            better = v > val[jmp]
            val[jmp[better]] = v
            b[jmp[better]] = j0
        delta = val + E[t]
        back[t] = b
    path = np.zeros(T, np.int64)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return np.array(seq)[path], float(delta.max())


def bass_chroma(feats: np.ndarray, bins_per_semitone: int = 2, octaves: int = 2, width: int = 9) -> np.ndarray:
    """Pitch-class energy of the lowest `octaves` (C2-B3): where the chord root usually sounds."""
    nb = 12 * bins_per_semitone * octaves
    x = feats[:, :nb]
    pcs = (np.arange(nb) // bins_per_semitone) % 12
    b = np.stack([x[:, pcs == pc].sum(1) for pc in range(12)], 1)
    k = np.ones(width, np.float32) / width
    b = np.stack([np.convolve(b[:, i], k, mode="same") for i in range(12)], 1)
    b = b - b.mean(1, keepdims=True)
    return b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)


def _root(c: int) -> int | None:
    d = decode_ext(c)
    return None if d is None else d[0]


def align_ordered(feats: np.ndarray, seq: list[int], bins_per_semitone: int = 2, frame_seconds: float = 0.04,
                  mean_chord_seconds: float = 2.0, beta: float = 8.0, shifts=range(12), jumps=None,
                  bass_weight: float = 0.0) -> dict:
    """Best transposition first; every shift's result is kept in "all" (for capo checks). The
    returned dict also carries the per-frame emission matrix ("cos", "states") for confidence."""
    ch = smooth_chroma(feats, bins_per_semitone)
    bch = bass_chroma(feats, bins_per_semitone) if bass_weight else None
    energy = feats.mean(1)
    quiet = energy < np.percentile(energy, 5) * 1.05
    p_stay = 1 - frame_seconds / mean_chord_seconds
    results = []
    for k in shifts:
        sq = [NO_CHORD] + [transpose(c, k) if c not in (NO_CHORD, IGNORE) else c for c in seq] + [NO_CHORD]
        states = sorted(set(sq))
        tpl = np.stack([template(c) for c in states])
        cos = ch @ tpl.T
        if bch is not None:
            roots = [_root(c) for c in states]
            cos = cos + bass_weight * np.stack([bch[:, r] if r is not None else np.zeros(len(ch)) for r in roots], 1)
        cos[:, states.index(NO_CHORD)] = np.where(quiet, 0.6, 0.15) * (1 + bass_weight)
        jj = None if jumps is None else [j + 1 for j in jumps]
        lab, ll = ordered_viterbi(cos, sq, states, p_stay, beta, jumps=jj)
        sidx = np.searchsorted(states, lab)
        on = lab != NO_CHORD
        fit = float(cos[np.arange(len(lab)), sidx][on].mean()) if on.any() else 0.0
        results.append({"shift": k, "labels": lab, "ll": ll, "fit": fit, "cos": cos, "states": states})
    order = sorted(range(len(results)), key=lambda i: -results[i]["ll"])
    best = dict(results[order[0]])
    best["margin"] = (best["ll"] - results[order[1]]["ll"]) / len(feats) if len(results) > 1 else None
    best["all"] = {r["shift"]: r for r in results}
    return best


def snap_to_beats(labels: np.ndarray, beat_frames: np.ndarray, radius: int = 6) -> np.ndarray:
    """Move every chord change to the nearest beat within +-radius frames (chords change on beats)."""
    out = labels.copy()
    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    if not len(beat_frames):
        return out
    for c in changes:
        j = np.searchsorted(beat_frames, c)
        cand = [b for b in beat_frames[max(0, j - 1): j + 1] if abs(b - c) <= radius]
        if not cand:
            continue
        b = min(cand, key=lambda x: abs(x - c))
        if b < c:
            out[b:c] = labels[c]
        elif b > c:
            out[c:b] = labels[c - 1]
    return out


def frame_confidence(labels: np.ndarray, cos: np.ndarray, states: list[int], width: int = 13) -> np.ndarray:
    """Per-frame margin: emission of the aligned chord minus the best *other* sheet chord, smoothed.
    Low = the audio doesn't clearly support the aligned chord there."""
    sidx = np.searchsorted(states, labels)
    own = cos[np.arange(len(labels)), sidx]
    other = cos.copy()
    other[np.arange(len(labels)), sidx] = -np.inf
    m = own - other.max(1)
    k = np.ones(width) / width
    return np.convolve(m, k, mode="same")
