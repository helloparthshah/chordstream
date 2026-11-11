"""Teacher -> student distillation data.

An offline, bidirectional teacher (`train --bidirectional`) sees future audio, so its
posteriors are both more accurate and *anticipatory* around chord changes. This module
writes a training cache where every song carries the teacher's posteriors ("soft"):

  * labelled songs keep their human labels and gain soft targets;
  * unlabelled audio (e.g. downloads that failed the alignment check) gets soft targets
    only, with labels set to IGNORE, so it trains the causal student through the
    distillation loss alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import FrontendConfig
from .cqt import offline_cqt
from .data.billboard import AUDIO_EXTS
from .data.dataset import load_audio
from .vocab import IGNORE


def pseudo_label(teacher: str, labelled: str, out: str, unlabelled_audio: list[str] | None = None,
                 unlabelled_caches: list[str] | None = None) -> dict:
    import torch

    from .export import load_net
    from .train import predict_song

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = load_net(Path(teacher))
    net.to(dev)
    if not cfg.model.bidirectional:
        print("warning: teacher is causal; pseudo-labels will not benefit from lookahead")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"labelled": 0, "unlabelled": 0}

    known = set()
    for f in sorted(Path(labelled).glob("*.npz")):
        d = np.load(f)
        soft = predict_song(net, d["feats"], dev)
        extra = {k: d[k] for k in ("root", "bass", "notes", "labels_mm") if k in d.files}  # structured targets
        np.savez_compressed(out_dir / f.name, feats=d["feats"], labels=d["labels"], soft=soft.astype(np.float16), **extra)
        known.add(f.stem.split("__")[-1])
        stats["labelled"] += 1

    # Songs that exist in another cache but were excluded from the labelled set (e.g. drifting
    # labels): reuse their features, drop their labels.
    for cache in unlabelled_caches or []:
        for f in sorted(Path(cache).glob("*.npz")):
            sid = f.stem.split("__")[-1]
            if sid in known or (out_dir / f"unlab__{sid}.npz").exists():
                continue
            feats = np.load(f)["feats"]
            soft = predict_song(net, feats, dev)
            np.savez_compressed(out_dir / f"unlab__{sid}.npz", feats=feats,
                                labels=np.full(len(feats), IGNORE, np.int16), soft=soft.astype(np.float16))
            stats["unlabelled"] += 1

    fcfg = cfg.frontend if cfg.frontend else FrontendConfig()
    for folder in unlabelled_audio or []:
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            dst = out_dir / f"unlab__{p.stem}.npz"
            if dst.exists():
                continue
            feats = offline_cqt(load_audio(p, fcfg.sample_rate), fcfg, dev).astype(np.float16)
            soft = predict_song(net, feats, dev)
            np.savez_compressed(dst, feats=feats, labels=np.full(len(feats), IGNORE, np.int16),
                                soft=soft.astype(np.float16))
            stats["unlabelled"] += 1
    return stats

