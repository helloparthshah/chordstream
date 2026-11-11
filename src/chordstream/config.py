"""Shared configuration. Frontend params are baked into every exported model bundle,
so training and live inference always see byte-identical features."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FrontendConfig:
    sample_rate: int = 16000
    hop_length: int = 640  # 40 ms frames at 16 kHz
    fmin: float = 65.406  # C2
    n_bins: int = 144  # 6 octaves: C2..B7
    bins_per_octave: int = 24  # 2 bins per semitone
    filter_scale: float = 1.0
    gamma: float = 3.0  # variable-Q offset (Hz); shortens low-frequency kernels -> less smearing
    log_scale: float = 100.0  # features = log1p(log_scale * |CQT|)

    @property
    def bins_per_semitone(self) -> int:
        return self.bins_per_octave // 12

    @property
    def frame_seconds(self) -> float:
        return self.hop_length / self.sample_rate


@dataclass
class ModelConfig:
    n_bins: int = 144
    n_classes: int = 25
    cnn_channels: tuple[int, ...] = (16, 32, 64)
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    context_frames: int = 128  # ~5.1 s of attention history used at inference


@dataclass
class BundleConfig:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BundleConfig":
        d = json.loads(Path(path).read_text())
        m = d["model"]
        m["cnn_channels"] = tuple(m["cnn_channels"])
        return cls(FrontendConfig(**d["frontend"]), ModelConfig(**m))
