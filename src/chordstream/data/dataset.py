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


TRAIN_ONLY = ("synthx__",)


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
                 n_classes: int = N_CLASSES):
        self.songs = []
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
                self.songs.append((d["feats"].astype(np.float32), d["labels"].astype(np.int64), soft, aux))
        if not self.songs:
            raise ValueError("no songs found for dataset")
        lengths = np.array([len(s[1]) for s in self.songs], dtype=np.float64)
        self.weights = lengths / lengths.sum()  # sample proportional to duration
        self.crop, self.bps, self.augment, self.n = crop, bins_per_semitone, augment, crops_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i, np.random.randint(1 << 30) if self.augment else 0))
        feats, labels, soft, aux_all = self.songs[rng.choice(len(self.songs), p=self.weights)]
        T = len(labels)
        q = np.zeros((self.crop, self.n_classes), np.float32)  # all-zero rows = no soft target
        aux = np.tile(np.array([IGNORE, IGNORE, -1], np.int64), (self.crop, 1))
        if T >= self.crop:
            s = int(rng.integers(0, T - self.crop + 1))
            x, y = feats[s : s + self.crop].copy(), labels[s : s + self.crop].copy()
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
            k = int(rng.integers(-5, 7))
            if k:
                if x.shape[1] > 144:  # E20: trailing 12 key columns rotate with the transposition
                    key = np.roll(x[:, 144:156], k, axis=1)
                    x = np.concatenate([shift_bins(x[:, :144], k * self.bps), key, x[:, 156:]], axis=1)
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
            x[:, :144] *= rng.uniform(0.8, 1.2)  # loudness jitter in log domain (CQT bins only)
            if rng.random() < 0.5:  # frequency masking
                f0 = int(rng.integers(0, 144 - 12))
                x[:, f0 : f0 + int(rng.integers(1, 12))] = 0
        return x, y, q, aux


def shift_bins(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(x)
    if n > 0:
        out[:, n:] = x[:, :-n]
    elif n < 0:
        out[:, :n] = x[:, -n:]
    else:
        out[:] = x
    return out
