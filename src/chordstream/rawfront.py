"""E33: a learned front end on the raw waveform instead of the CQT.

RawFrontend is a causal strided Conv1d over 16 kHz audio: 2 x 144 filters of `kernel` samples
(default 4096 = 256 ms, enough to separate semitones at C2), stride = hop (640 = 40 ms). Filter
pairs are combined as a magnitude sqrt(a^2 + b^2) (a learned quadrature pair, like the CQT's complex
kernels) and compressed with log1p(scale * mag), scale learned. Frame t uses only the `kernel`
samples ending at the end of frame t, exactly like the causal CQT, so it is a drop-in replacement for
ChordNet's 144-bin input.

RawCropDataset mirrors CropDataset for raw audio: random crops of `crop` frames (+ the front end's
left context) with labels, and pitch-shift augmentation by resampling (k semitones = resample by
2^(k/12), labels time-stretched to match and transposed).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import FrontendConfig
from .data.dataset import _TRANSPOSE_TABLE, load_audio, rotate_mask
from .model import ChordNet
from .vocab import IGNORE

RAW_DIR = Path("data/raw16k")


class RawFrontend(nn.Module):
    def __init__(self, n_bins: int = 144, kernel: int = 4096, hop: int = 640):
        super().__init__()
        self.kernel, self.hop, self.n_bins = kernel, hop, n_bins
        self.conv = nn.Conv1d(1, 2 * n_bins, kernel, stride=hop, bias=False)
        nn.init.normal_(self.conv.weight, std=1.0 / math.sqrt(kernel))
        self.log_scale = nn.Parameter(torch.tensor(math.log(100.0)))

    @property
    def context(self) -> int:
        """Samples of audio before the first frame's hop that frame 0 needs."""
        return self.kernel - self.hop

    def forward(self, audio: torch.Tensor) -> torch.Tensor:  # (B, context + T*hop) -> (B, T, n_bins)
        z = self.conv(audio.unsqueeze(1).float())  # (B, 2F, T)
        a, b = z[:, : self.n_bins], z[:, self.n_bins :]
        mag = torch.sqrt(a * a + b * b + 1e-10)
        return torch.log1p(self.log_scale.exp() * mag).transpose(1, 2)


class RawChordNet(nn.Module):
    def __init__(self, net: ChordNet, kernel: int = 4096, hop: int = 640):
        super().__init__()
        self.front = RawFrontend(net.cfg.n_bins, kernel, hop)
        self.net = net
        self.cfg = net.cfg

    def forward_all(self, audio: torch.Tensor) -> dict:
        with torch.autocast(audio.device.type, enabled=False):
            feats = self.front(audio)
        return self.net.forward_all(feats)


def prepare_raw(stem: str, audio_path: Path, sr: int = 16000) -> Path:
    """Cache a song's 16 kHz mono audio as int16 .npy (memory-mapped during training)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = RAW_DIR / f"{stem}.npy"
    if not dst.exists():
        y = load_audio(audio_path, sr)
        np.save(dst, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    return dst


class RawCropDataset(torch.utils.data.Dataset):
    """items: list of (raw .npy path, label .npz path)."""

    def __init__(self, items, crop: int, context: int, hop: int = 640, crops_per_epoch: int = 2000,
                 augment: bool = True, seed: int = 0):
        self.songs = []
        for raw, lab in items:
            d = np.load(lab)
            aux = np.stack([d["root"], d["bass"], d["notes"]], 1).astype(np.int64)
            self.songs.append((np.load(raw, mmap_mode="r"), d["labels"].astype(np.int64), aux))
        lengths = np.array([len(s[1]) for s in self.songs], np.float64)
        self.p = lengths / lengths.sum()
        self.crop, self.context, self.hop, self.n, self.augment, self.seed = crop, context, hop, crops_per_epoch, augment, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, np.random.randint(1 << 30)))
        audio, labels, aux_all = self.songs[rng.choice(len(self.songs), p=self.p)]
        k = int(rng.integers(-5, 7)) if self.augment else 0
        r = 2.0 ** (k / 12)  # play r times faster: pitch up k semitones
        n_out = self.context + self.crop * self.hop
        n_src = int(math.ceil(n_out * r)) + 2
        T = len(labels)
        # choose the source start (in samples) so the crop's frames fall inside the song
        last_start = max(1, T * self.hop - n_src)
        s0 = int(rng.integers(-self.context, last_start))
        idx = np.arange(s0, s0 + n_src)
        seg = np.zeros(n_src, np.float32)
        ok = (idx >= 0) & (idx < len(audio))
        seg[ok] = audio[idx[ok]].astype(np.float32) / 32767.0
        x = np.interp(np.arange(n_out) * r, np.arange(n_src), seg).astype(np.float32) if k else seg[:n_out].copy()
        if self.augment:
            x *= rng.uniform(0.5, 2.0)
        # label of output frame f: the source frame whose end sample matches the output frame's end
        end_out = self.context + (np.arange(self.crop) + 1) * self.hop
        src_frame = np.floor((s0 + end_out * r) / self.hop).astype(np.int64) - 1
        valid = (src_frame >= 0) & (src_frame < T)
        y = np.full(self.crop, IGNORE, np.int64)
        aux = np.tile(np.array([IGNORE, IGNORE, -1], np.int64), (self.crop, 1))
        y[valid] = labels[src_frame[valid]]
        aux[valid] = aux_all[src_frame[valid]]
        if k:
            v = y >= 0
            y[v] = _TRANSPOSE_TABLE[k % 12][y[v]]
            for col in (0, 1):
                vv = (aux[:, col] >= 0) & (aux[:, col] < 12)
                aux[vv, col] = (aux[vv, col] + k) % 12
            aux[:, 2] = rotate_mask(aux[:, 2], k)
        q = np.zeros((self.crop, 1), np.float32)  # no teacher posteriors
        return x, y, q, aux


@torch.no_grad()
def raw_features(model: RawChordNet, audio: np.ndarray, device, chunk_frames: int = 2048) -> np.ndarray:
    """Whole-song front-end features (exactly causal), computed in chunks."""
    model.eval()
    hop, ctx = model.front.hop, model.front.context
    T = len(audio) // hop
    padded = np.concatenate([np.zeros(ctx, np.float32), audio.astype(np.float32)])
    out = []
    for f0 in range(0, T, chunk_frames):
        f1 = min(T, f0 + chunk_frames)
        seg = torch.from_numpy(padded[f0 * hop : ctx + f1 * hop]).to(device)[None]
        out.append(model.front(seg)[0].cpu().numpy())
    return np.concatenate(out)[:T]


def frontend_config() -> FrontendConfig:
    return FrontendConfig()
