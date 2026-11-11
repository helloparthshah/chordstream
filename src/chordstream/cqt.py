"""Causal (variable-)Constant-Q transform.

Every kernel is right-aligned to the *current* sample: the frame emitted at time t
uses only audio in [t - L_k, t]. No lookahead, no centering.

The kernel bank is built in numpy so the streaming engine (numpy + onnxruntime,
no torch) and the training pipeline (torch, on GPU) compute identical features.
"""

from __future__ import annotations

import numpy as np

from .config import FrontendConfig


def cqt_frequencies(cfg: FrontendConfig) -> np.ndarray:
    return cfg.fmin * 2.0 ** (np.arange(cfg.n_bins) / cfg.bins_per_octave)


def kernel_lengths(cfg: FrontendConfig) -> np.ndarray:
    alpha = 2.0 ** (1.0 / cfg.bins_per_octave) - 1.0
    freqs = cqt_frequencies(cfg)
    return np.ceil(cfg.filter_scale * cfg.sample_rate / (alpha * freqs + cfg.gamma)).astype(int)


def build_kernels(cfg: FrontendConfig) -> np.ndarray:
    """Complex kernel bank of shape (n_bins, L_max), each right-aligned."""
    freqs = cqt_frequencies(cfg)
    lengths = kernel_lengths(cfg)
    L = int(lengths.max())
    K = np.zeros((cfg.n_bins, L), dtype=np.complex64)
    for k, (f, n) in enumerate(zip(freqs, lengths)):
        t = np.arange(n) - (n - 1)  # ends at 0 == "now"
        win = np.hanning(n)
        ker = win * np.exp(2j * np.pi * f * t / cfg.sample_rate)
        K[k, L - n:] = ker / win.sum()  # unit gain for a full-scale sinusoid (x2 for one-sided)
    return K * 2.0


def compress(mag: np.ndarray, cfg: FrontendConfig) -> np.ndarray:
    return np.log1p(cfg.log_scale * mag)


class StreamingCQT:
    """Numpy single-frame CQT for live use: feed audio, get one frame per hop."""

    def __init__(self, cfg: FrontendConfig):
        self.cfg = cfg
        self.K = build_kernels(cfg)
        self.L = self.K.shape[1]
        self.buf = np.zeros(self.L, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)

    def push(self, audio: np.ndarray) -> np.ndarray:
        """Append audio; return (n_new_frames, n_bins) features."""
        hop = self.cfg.hop_length
        self.pending = np.concatenate([self.pending, audio.astype(np.float32)])
        frames = []
        while len(self.pending) >= hop:
            chunk, self.pending = self.pending[:hop], self.pending[hop:]
            self.buf = np.concatenate([self.buf[hop:], chunk])
            frames.append(np.abs(self.K @ self.buf))
        if not frames:
            return np.zeros((0, self.cfg.n_bins), dtype=np.float32)
        return compress(np.stack(frames), self.cfg).astype(np.float32)


def offline_cqt(audio: np.ndarray, cfg: FrontendConfig, device: str | None = None) -> np.ndarray:
    """Whole-file causal CQT (identical to StreamingCQT frame-for-frame). Uses torch if available.

    Frame i covers audio up to sample (i + 1) * hop, matching StreamingCQT's i-th output.
    """
    K = build_kernels(cfg)
    L, hop = K.shape[1], cfg.hop_length
    n_frames = len(audio) // hop
    x = np.concatenate([np.zeros(L - hop, dtype=np.float32), audio[: n_frames * hop].astype(np.float32)])
    try:
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        w = torch.from_numpy(np.concatenate([K.real, K.imag]).astype(np.float32)).to(dev)[:, None, :]
        xt = torch.from_numpy(x).to(dev)[None, None, :]
        out = []
        step = 4096  # frames per chunk keeps memory bounded on long songs
        with torch.no_grad():
            for s in range(0, n_frames, step):
                e = min(n_frames, s + step)
                seg = xt[..., s * hop : (e - 1) * hop + L]
                y = torch.nn.functional.conv1d(seg, w, stride=hop)[0]
                out.append(torch.sqrt(y[: cfg.n_bins] ** 2 + y[cfg.n_bins :] ** 2).T.cpu().numpy())
        mag = np.concatenate(out) if out else np.zeros((0, cfg.n_bins), np.float32)
    except ImportError:
        idx = np.arange(n_frames)[:, None] * hop + np.arange(L)[None, :]
        mag = np.abs(x[idx] @ K.T)
    return compress(mag, cfg).astype(np.float32)
