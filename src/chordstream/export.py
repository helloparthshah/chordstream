"""Export a trained bundle to an incremental (KV-cached) ONNX step model.

Writes into the bundle:
  model.onnx      one 40 ms step: (frames[1,RF,F], k_cache, v_cache) -> (probs, k_cache, v_cache)
  init_state.npz  KV caches after a long stretch of silence: the stream's start state
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import BundleConfig
from .model import ChordNet, StreamingStep, receptive_field_cnn


def load_net(bundle_dir: Path) -> tuple[ChordNet, BundleConfig]:
    cfg = BundleConfig.load(bundle_dir / "config.json")
    net = ChordNet(cfg.model)
    net.load_state_dict(torch.load(bundle_dir / "model.pt", map_location="cpu")["state_dict"])
    return net.eval(), cfg


@torch.no_grad()
def silence_state(step: StreamingStep, net: ChordNet, cfg: BundleConfig):
    rf = receptive_field_cnn(cfg.model)
    frames = torch.zeros(1, rf, cfg.model.n_bins)
    k = torch.zeros(net.cache_shape())
    v = torch.zeros(net.cache_shape())
    for _ in range(cfg.model.n_layers * cfg.model.context_frames):  # flush every layer's window
        _, k, v = step(frames, k, v)
    return k, v


def export_onnx(bundle_dir: str | Path, opset: int = 17) -> Path:
    bundle_dir = Path(bundle_dir)
    net, cfg = load_net(bundle_dir)
    step = StreamingStep(net).eval()
    rf = receptive_field_cnn(cfg.model)
    k0, v0 = silence_state(step, net, cfg)
    np.savez(bundle_dir / "init_state.npz", k=k0.numpy(), v=v0.numpy())

    frames = torch.rand(1, rf, cfg.model.n_bins) * 3
    path = bundle_dir / "model.onnx"
    torch.onnx.export(
        step, (frames, k0, v0), str(path), opset_version=opset, dynamo=False,
        input_names=["frames", "k_cache", "v_cache"], output_names=["probs", "k_out", "v_out"],
    )

    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = step(frames, k0, v0)
    got = sess.run(None, {"frames": frames.numpy(), "k_cache": k0.numpy(), "v_cache": v0.numpy()})
    err = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
    if err > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max abs err {err}")
    print(f"exported {path}  (parity max err {err:.2e})")
    return path
