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
        _, k, v = step(frames, k, v)[:3]
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
        input_names=["frames", "k_cache", "v_cache"],
        output_names=["probs", "k_out", "v_out"] + (["sheet_probs"] if cfg.model.sheet_delay else []),
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


def export_web(bundle_dir: str | Path, out_dir: str | Path) -> Path:
    """Write a bundle in the browser engine's format (web/engine.js): the ONNX step model, the
    silence start state and the HMM as little-endian raw arrays, and a meta.json describing them.
    The CQT kernels are rebuilt in the browser from the frontend config (identical formula)."""
    import json

    from .hmm import OnlineHMM

    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = BundleConfig.load(bundle_dir / "config.json")
    st = np.load(bundle_dir / "init_state.npz")
    k, v = st["k"].astype("<f4"), st["v"].astype("<f4")
    # Re-export the step model (StreamingStep now computes only the newest frame's CNN cone).
    net, _ = load_net(bundle_dir)
    step = StreamingStep(net).eval()
    frames = torch.rand(1, receptive_field_cnn(cfg.model), cfg.model.n_bins) * 3
    k0, v0 = torch.from_numpy(st["k"]), torch.from_numpy(st["v"])
    torch.onnx.export(step, (frames, k0, v0), str(out_dir / "model.onnx"), opset_version=17, dynamo=False,
                      input_names=["frames", "k_cache", "v_cache"], output_names=["probs", "k_out", "v_out"])
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = step(frames, k0, v0)
    got = sess.run(None, {"frames": frames.numpy(), "k_cache": st["k"], "v_cache": st["v"]})
    err = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
    if err > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max abs err {err}")
    (out_dir / "state.bin").write_bytes(k.tobytes() + v.tobytes())
    hmm = OnlineHMM.load(bundle_dir / "hmm.npz")
    (out_dir / "hmm.bin").write_bytes(hmm.A.astype("<f8").tobytes() + hmm.prior.astype("<f8").tobytes())
    meta = {
        "frontend": cfg.frontend.__dict__,
        "rf": receptive_field_cnn(cfg.model),
        "n_bins": cfg.model.n_bins,
        "n_classes": cfg.model.n_classes,
        "label_delay": cfg.model.label_delay,
        "context_frames": cfg.model.context_frames,
        "cache_shape": list(k.shape),
        "hmm_power": float(hmm.power),
        "bundle": bundle_dir.name,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"web bundle -> {out_dir}  (parity {err:.1e}, {sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6:.1f} MB)")
    return out_dir
