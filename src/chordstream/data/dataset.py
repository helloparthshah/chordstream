"""Feature caching and the training dataset (random causal crops + pitch-shift augmentation)."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import FrontendConfig
from ..cqt import offline_cqt
from ..vocab import IGNORE, N_CLASSES, N_EXT, NO_CHORD, NO_ROOT, parse_harte, parse_harte_ext, to_majmin, transpose
from .billboard import find_pairs, read_lab

# Covers the extended vocabulary; its first 25 columns are the majmin table (closed under transposition).
_TRANSPOSE_TABLE = np.array(
    [[transpose(c, k) for c in range(N_EXT)] for k in range(12)], dtype=np.int64
)


def rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """Rotate 12-bit pitch-class masks up by k semitones (-1 = unknown is kept)."""
    k %= 12
    m = mask.astype(np.int64)
    out = ((m << k) | (m >> (12 - k))) & 0xFFF
    return np.where(mask < 0, -1, out)


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        import soxr

        y = soxr.resample(y, file_sr, sr).astype(np.float32)
    return y


def frame_labels(segs, n_frames: int, cfg: FrontendConfig) -> np.ndarray:
    """Label frame i with the chord sounding at its newest audio (the moment it is emitted)."""
    times = (np.arange(n_frames) + 0.5) * cfg.frame_seconds
    labels = np.full(n_frames, NO_CHORD, dtype=np.int16)
    for s, e, c in segs:
        labels[(times >= s) & (times < e)] = parse_harte(c)
    return labels


def frame_labels_ext(segs, n_frames: int, cfg: FrontendConfig):
    """Extended labels per frame: (class, root, bass, note mask) - see vocab.parse_harte_ext."""
    times = (np.arange(n_frames) + 0.5) * cfg.frame_seconds
    labels = np.full(n_frames, NO_CHORD, np.int16)
    root = np.full(n_frames, NO_ROOT, np.int16)
    bass = np.full(n_frames, NO_ROOT, np.int16)
    notes = np.zeros(n_frames, np.int32)
    for s, e, c in segs:
        m = (times >= s) & (times < e)
        labels[m], root[m], bass[m], notes[m] = parse_harte_ext(c)
    return labels, root, bass, notes


def build_extended_cache(lab_dir: str | Path, cache_orig: str | Path, cache_clean: str | Path, out: str | Path) -> dict:
    """Extended-label twin of the cleaned cache: same songs, same features, same re-timing.

    The cleaned cache re-timed some songs' majmin labels (data/audit.realign). The re-timing is
    deterministic, so it is recomputed here from the original features and labels and applied to
    the extended arrays; songs whose recomputed timing does not reproduce the cleaned labels exactly
    are skipped rather than risk mismatched labels.
    """
    from .audit import apply_map, realign_map
    from .billboard import read_lab

    cfg = FrontendConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "retimed": 0, "skipped": 0}
    for f in sorted(Path(cache_clean).glob("*.npz")):
        sid = f.stem.split("__")[-1]
        clean = np.load(f)
        orig = np.load(Path(cache_orig) / f.name)
        feats, mm_clean, mm_orig = clean["feats"], clean["labels"].astype(np.int64), orig["labels"].astype(np.int64)
        ext = frame_labels_ext(read_lab(Path(lab_dir) / f"{sid}.lab"), len(mm_orig), cfg)
        m = None
        if not np.array_equal(mm_clean, mm_orig):
            m = realign_map(orig["feats"].astype(np.float32), mm_orig)
            if m is None or not np.array_equal(apply_map(mm_orig, m, NO_CHORD), mm_clean):
                stats["skipped"] += 1
                continue
            stats["retimed"] += 1
        labels, root, bass, notes = (apply_map(a, m, fill) for a, fill in zip(ext, (NO_CHORD, NO_ROOT, NO_ROOT, 0)))
        np.savez_compressed(out / f.name, feats=feats, labels=labels.astype(np.int16), root=root.astype(np.int16),
                            bass=bass.astype(np.int16), notes=notes.astype(np.int32), labels_mm=mm_clean.astype(np.int16))
        stats["written"] += 1
    return stats


def cache_features(src_dirs: list[str | Path], out: str | Path, cfg: FrontendConfig, device: str | None = None) -> int:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in src_dirs:
        for audio_path, lab_path in find_pairs(d):
            dst = out / f"{Path(d).name}__{audio_path.stem}.npz"
            if dst.exists():
                continue
            y = load_audio(audio_path, cfg.sample_rate)
            feats = offline_cqt(y, cfg, device)
            labels = frame_labels(read_lab(lab_path), len(feats), cfg)
            np.savez_compressed(dst, feats=feats.astype(np.float16), labels=labels)
            n += 1
    return n


TRAIN_ONLY = ("synthx__", "bollytrain__", "roomx__", "tunex__", "choco__", "bbrec__")  # synthetic renders; Bollywood training copies (E21); room-echo copies (E29)


def cache_features_ext(src: str | Path, out: str | Path, prefix: str, cfg: FrontendConfig | None = None,
                       device: str | None = None) -> int:
    """Extended-label cache (same arrays as data/cache_ext) for rendered songs with .lab files."""
    cfg = cfg or FrontendConfig()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for audio_path, lab_path in find_pairs(src):
        dst = out / f"{prefix}__{audio_path.stem}.npz"
        if dst.exists():
            continue
        feats = offline_cqt(load_audio(audio_path, cfg.sample_rate), cfg, device)
        labels, root, bass, notes = frame_labels_ext(read_lab(lab_path), len(feats), cfg)
        mm = np.array([to_majmin(int(c)) if c >= 0 else c for c in labels], np.int16)
        np.savez_compressed(dst, feats=feats.astype(np.float16), labels=labels, root=root, bass=bass, notes=notes,
                            labels_mm=mm)
        n += 1
    return n


def split_files(cache: str | Path, val_fraction: float = 0.1, holdout_prefix: str | None = None):
    files = sorted(Path(cache).glob("*.npz"))
    train, val = [], []
    for f in files:
        if f.name.startswith(TRAIN_ONLY):  # synthetic songs never enter the held-out set
            is_val = False
        elif holdout_prefix is not None:
            is_val = f.name.startswith(holdout_prefix)
        else:
            is_val = (zlib.crc32(f.stem.encode()) % 1000) / 1000 < val_fraction
        (val if is_val else train).append(f)
    return train, val


class CropDataset:
    """Map-style dataset of random crops: (feats[T,F], labels[T], soft[T,C], aux[T,3]).

    soft: teacher posteriors (all-zero rows = none). aux: root, bass (0..11, NO_ROOT=12 for N,
    IGNORE unknown) and 12-bit note mask (-1 unknown), present in extended-label caches.
    torch-agnostic numpy output.
    """

    def __init__(self, files, crop: int, bins_per_semitone: int, augment: bool, crops_per_epoch: int, seed: int = 0,
                 n_classes: int = N_CLASSES, bass_dropout: float = 0.0, random_eq: float = 0.0,
                 micro_shift: bool = False, tune_share: float = 0.0, shares: dict[str, float] | None = None,
                 beat: bool = False):
        self.songs = []
        self.beats = [] if beat else None  # E42: causal beat features per song, appended after augmentation
        is_tune, prefixes = [], []
        self.n_classes = n_classes
        for f in files:
            d = np.load(f)
            if len(d["labels"]) >= 8:
                soft = d["soft"].astype(np.float32) if "soft" in d.files else None  # teacher posteriors
                T = len(d["labels"])
                if "root" in d.files:
                    aux = np.stack([d["root"], d["bass"], d["notes"]], 1).astype(np.int64)
                else:
                    aux = np.tile(np.array([IGNORE, IGNORE, -1], np.int64), (T, 1))
                feats = d["feats"]  # kept as stored (float16): crops are converted on read, halving RAM
                self.songs.append((feats if feats.dtype == np.float16 else feats.astype(np.float32),
                                   d["labels"].astype(np.int64), soft, aux))
                if beat:
                    from ..beat import cached_beat_features
                    self.beats.append(cached_beat_features(f, d["feats"]))
                is_tune.append(Path(f).name.startswith("tunex__"))
                prefixes.append(Path(f).name.split("__")[0])
        if not self.songs:
            raise ValueError("no songs found for dataset")
        lengths = np.array([len(s[1]) for s in self.songs], dtype=np.float64)
        self.weights = lengths / lengths.sum()  # sample proportional to duration
        tune = np.array(is_tune, bool)
        if tune_share > 0 and tune.any() and (~tune).any():  # E35b: detuned copies get this share of the crops
            self.weights[tune] *= tune_share / self.weights[tune].sum()
            self.weights[~tune] *= (1 - tune_share) / self.weights[~tune].sum()
        if shares:  # E40b: fixed share of crops per file prefix (e.g. bollytrain=0.52); the rest by duration
            pre = np.array(prefixes)
            fixed = np.zeros(len(pre), bool)
            for prefix, frac in shares.items():
                sel = pre == prefix
                if sel.any():
                    self.weights[sel] *= frac / self.weights[sel].sum()
                    fixed |= sel
            rest = 1 - sum(f for p_, f in shares.items() if (pre == p_).any())
            if (~fixed).any():
                self.weights[~fixed] *= rest / self.weights[~fixed].sum()
            self.weights /= self.weights.sum()
        self.crop, self.bps, self.augment, self.n = crop, bins_per_semitone, augment, crops_per_epoch
        self.seed = seed
        self.bass_dropout = bass_dropout  # E31: share of crops with the low register removed
        self.random_eq = random_eq  # E32: share of crops passed through a random EQ curve
        self.micro_shift = micro_shift  # E37: random +-1 sub-semitone bin shift, labels unchanged

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i, np.random.randint(1 << 30) if self.augment else 0))
        j = rng.choice(len(self.songs), p=self.weights)
        feats, labels, soft, aux_all = self.songs[j]
        T = len(labels)
        s = 0
        q = np.zeros((self.crop, self.n_classes), np.float32)  # all-zero rows = no soft target
        aux = np.tile(np.array([IGNORE, IGNORE, -1], np.int64), (self.crop, 1))
        if T >= self.crop:
            s = int(rng.integers(0, T - self.crop + 1))
            x, y = feats[s : s + self.crop].astype(np.float32), labels[s : s + self.crop].copy()
            aux[:] = aux_all[s : s + self.crop]
            if soft is not None:
                q[:] = soft[s : s + self.crop]
        else:
            x = np.zeros((self.crop, feats.shape[1]), np.float32)
            y = np.full(self.crop, IGNORE, np.int64)
            x[-T:], y[-T:], aux[-T:] = feats, labels, aux_all
            if soft is not None:
                q[-T:] = soft
        if self.augment:
            nb = 72 * self.bps  # CQT bins (6 octaves); any columns after them are extra inputs
            k = int(rng.integers(-5, 7))
            if k:
                if x.shape[1] > nb:  # E20: trailing 12 key columns rotate with the transposition
                    key = np.roll(x[:, nb:nb + 12], k, axis=1)
                    x = np.concatenate([shift_bins(x[:, :nb], k * self.bps), key, x[:, nb + 12:]], axis=1)
                else:
                    x = shift_bins(x, k * self.bps)
                valid = y >= 0
                y[valid] = _TRANSPOSE_TABLE[k % 12][y[valid]]
                moved = np.zeros_like(q)
                moved[:, _TRANSPOSE_TABLE[k % 12][: q.shape[1]]] = q  # class c's mass moves to c transposed by k
                q = moved
                for col in (0, 1):  # root, bass: shift real pitch classes only
                    v = (aux[:, col] >= 0) & (aux[:, col] < 12)
                    aux[v, col] = (aux[v, col] + k) % 12
                aux[:, 2] = rotate_mask(aux[:, 2], k)
            if self.micro_shift and self.bps >= 3:  # E37: label-preserving +-1 sub-semitone bin (+-33 cents at 3/semitone)
                j = int(rng.integers(-1, 2))
                if j:
                    x[:, :nb] = shift_bins(x[:, :nb], j)
            x[:, :nb] *= rng.uniform(0.8, 1.2)  # loudness jitter in log domain (CQT bins only)
            if self.bass_dropout and rng.random() < self.bass_dropout:
                # E31 bass dropout: remove (or strongly attenuate) everything below a random cutoff,
                # 6-30 semitones above C2 (F#2 ~92 Hz .. F#4 ~370 Hz), with a 2-semitone ramp, so the
                # chord must be read from the upper harmony rather than the bass line.
                cut = int(rng.integers(6, 31)) * self.bps
                ramp = np.clip((np.arange(nb) - (cut - 2 * self.bps)) / (2 * self.bps), 0, 1)
                floor = rng.uniform(0.0, 0.3)
                x[:, :nb] *= (floor + (1 - floor) * ramp).astype(np.float32)
            if self.random_eq and rng.random() < self.random_eq:
                x[:, :nb] = apply_gain_db(x[:, :nb], random_eq_db(rng, self.bps, nb))
            if rng.random() < 0.5:  # frequency masking
                f0 = int(rng.integers(0, nb - 6 * self.bps))
                x[:, f0 : f0 + int(rng.integers(1, 6 * self.bps))] = 0
        if self.beats is not None:  # beat phase doesn't move with pitch shifts or EQ: append after augmenting
            b = np.zeros((self.crop, self.beats[j].shape[1]), np.float32)
            if T >= self.crop:
                b[:] = self.beats[j][s : s + self.crop]
            else:
                b[-T:] = self.beats[j]
            x = np.concatenate([x, b], axis=1)
        return x, y, q, aux


def random_eq_db(rng, bps: int = 2, n_bins: int = 144) -> np.ndarray:
    """E32: a random EQ curve over the CQT bins (C2..B7), in dB: low shelf (cut up to 40 dB or
    boost 6 dB, corner 70-400 Hz), high shelf (-24..+6 dB, corner 1-3.5 kHz), a spectral tilt of
    +-6 dB, and 0-3 peaking/notch bands (-15..+10 dB, 1-12 semitones wide). Covers thin phone
    speakers, missing or boomy bass, muffled rooms and resonances."""
    st = np.arange(n_bins) / bps  # semitones above C2 (65.4 Hz)
    hz2st = lambda f: 12 * np.log2(f / 65.41)  # noqa: E731
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    g = np.zeros(n_bins)
    if rng.random() < 0.7:
        lo = rng.uniform(-40, 6) if rng.random() < 0.85 else rng.uniform(0, 6)
        g += lo * (1 - sig((st - hz2st(rng.uniform(70, 400))) / rng.uniform(0.7, 3)))
    if rng.random() < 0.5:
        g += rng.uniform(-24, 6) * sig((st - hz2st(rng.uniform(1000, 3500))) / rng.uniform(0.7, 3))
    g += rng.uniform(-6, 6) * (st / st[-1] - 0.5)
    for _ in range(int(rng.integers(0, 4))):
        g += rng.uniform(-15, 10) * np.exp(-0.5 * ((st - rng.uniform(0, st[-1])) / rng.uniform(0.5, 6)) ** 2)
    return g


def apply_gain_db(x: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """Exact magnitude gain on log-compressed CQT features (x = log1p(log_scale * |X|))."""
    gl = (10.0 ** (gain_db / 20.0)).astype(np.float32)
    return np.log1p(np.expm1(x) * gl).astype(np.float32)


def shift_bins(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(x)
    if n > 0:
        out[:, n:] = x[:, :-n]
    elif n < 0:
        out[:, :n] = x[:, -n:]
    else:
        out[:] = x
    return out
