"""F06: does live-like audio (room echo, phone mic, background noise) make the readout flicker more?

Re-renders the 36 clean Billboard test songs with simulated degradations, recomputes the causal CQT,
and scores the live HMM path (same labels). Degradations (seeded, per song):
  room    convolution with a synthetic room impulse response (exponentially decaying noise, RT60 0.5 s,
          plus a direct path), i.e. a speaker heard across a room
  phone   phone-mic band (smooth 2nd-order-like high-pass 150 Hz, low-pass 6 kHz, applied in the FFT domain) and soft clipping
  noise   pink noise at 15 dB SNR (room chatter / fan)
  all     room + phone + noise

usage: python experiments/analysis_degraded.py models/v9
"""
import sys
from pathlib import Path

import numpy as np
import torch

from chordstream.config import FrontendConfig
from chordstream.cqt import offline_cqt
from chordstream.data.dataset import load_audio, split_files
from chordstream.export import load_net
from chordstream.hmm import OnlineHMM
from chordstream.train import predict_song

sys.path.insert(0, str(Path(__file__).parent))
from analysis_flicker_fixes import stats  # noqa: E402


def rir(sr, rt60, rng):
    n = int(sr * rt60 * 1.2)
    t = np.arange(n) / sr
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    h[: int(0.003 * sr)] = 0
    h[0] = 1.0 / 0.4  # direct path ~ 8 dB above the reverberant tail energy
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)


def fftconvolve(x, h):
    n = len(x) + len(h) - 1
    m = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(h, m), m)[:n]


def band(x, sr, lo, hi):
    f = np.fft.rfftfreq(len(x), 1 / sr)
    g = (f / lo) ** 2 / (1 + (f / lo) ** 2) / (1 + (f / hi) ** 4)  # |H| of 2nd-order HP x LP, roughly
    return np.fft.irfft(np.fft.rfft(x) * np.sqrt(g), len(x))


def pink(n, rng):
    f = np.fft.rfft(rng.standard_normal(n))
    f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
    x = np.fft.irfft(f, n)
    return x / x.std()


def degrade(y, sr, kind, rng):
    out = y.copy()
    if kind in ("room", "all"):
        out = fftconvolve(out, rir(sr, 0.5, rng))[: len(y)]
    if kind in ("phone", "all"):
        out = band(out, sr, 150, 6000)
        out = np.tanh(2 * out / (np.abs(out).max() + 1e-9)) / 2
    if kind in ("noise", "all"):
        out = out + pink(len(out), rng) * out.std() / 10 ** (15 / 20)
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def main(b):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = FrontendConfig()
    net, _ = load_net(Path(b)); net.to(dev); hmm = OnlineHMM.load(Path(b) / "hmm.npz")
    _, val = split_files("data/cache_ext")
    for kind in ("clean", "room", "phone", "noise", "all"):
        pairs = []
        for f in val:
            d = np.load(f); y = d["labels"].astype(np.int64)
            audio = load_audio(Path("data/billboard") / f"{f.stem.split('__')[-1]}.flac", cfg.sample_rate)
            if kind != "clean":
                audio = degrade(audio, cfg.sample_rate, kind, np.random.default_rng(zlib_seed(f.stem)))
            feats = offline_cqt(audio, cfg, dev)
            n = min(len(feats), len(y))
            pairs.append((hmm.filter(predict_song(net, feats[:n].astype(np.float32), dev)), y[:n]))
        print(f"{kind:6s} {stats(pairs)}", flush=True)


def zlib_seed(s):
    import zlib
    return zlib.crc32(s.encode())


if __name__ == "__main__":
    main(sys.argv[1])
