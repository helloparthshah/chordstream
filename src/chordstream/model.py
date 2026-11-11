"""CNN + causal Transformer chord recognizer.

Input  : (B, T, F) log-CQT frames
Output : (B, T, n_classes) logits; output[t] depends only on frames <= t.

Causality is enforced in both halves:
  * CNN: time-axis padding is left-only, pooling happens only along frequency.
  * Transformer: banded causal mask (each frame attends to itself + the previous
    context_frames - 1 frames) with an ALiBi distance bias. No absolute positions, so
    the model is shift-invariant in time.

Streaming (StreamingStep) computes exactly the same function incrementally: the CNN
runs on only its 13-frame receptive field and each layer keeps a sliding KV cache of
context_frames - 1 entries, so one 40 ms step costs one token's worth of compute.
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class CausalConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, t_kernel: int = 3, f_kernel: int = 5, centered: bool = False):
        super().__init__()
        self.t_pad = t_kernel - 1
        # Causal: all padding on the past side. Centred (offline teacher): split evenly.
        self.pads = (self.t_pad // 2, self.t_pad - self.t_pad // 2) if centered else (self.t_pad, 0)
        self.conv1 = nn.Conv2d(cin, cout, (t_kernel, f_kernel), padding=(0, f_kernel // 2))
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, (t_kernel, f_kernel), padding=(0, f_kernel // 2))
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool2d((1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T, F)
        h = F.gelu(self.bn1(self.conv1(F.pad(x, (0, 0, *self.pads)))))
        h = self.bn2(self.conv2(F.pad(h, (0, 0, *self.pads))))
        return self.pool(F.gelu(h + self.skip(x)))


def receptive_field_cnn(cfg: ModelConfig) -> int:
    return 1 + 2 * 2 * len(cfg.cnn_channels)  # two time-kernel-3 convs per block


def alibi_slopes(n_heads: int) -> torch.Tensor:
    return torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)])


class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.dropout = dropout

    def _qkv(self, x):
        B, T, D = x.shape
        return self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self._qkv(x)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, D))

    def step(self, x, k_cache, v_cache, bias):
        """x: (B,1,D); caches: (B,H,W-1,dh); bias: (1,H,1,W). Returns y, new caches."""
        B, _, D = x.shape
        q, k, v = self._qkv(x)
        K = torch.cat([k_cache, k], dim=2)
        V = torch.cat([v_cache, v], dim=2)
        y = F.scaled_dot_product_attention(q, K, V, attn_mask=bias)
        return self.out(y.transpose(1, 2).reshape(B, 1, D)), K[:, :, 1:], V[:, :, 1:]


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * cfg.ff_mult), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d * cfg.ff_mult, d)
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, bias):
        x = x + self.drop(self.attn(self.ln1(x), bias))
        return x + self.drop(self.ff(self.ln2(x)))

    def step(self, x, kc, vc, bias):
        a, kc, vc = self.attn.step(self.ln1(x), kc, vc, bias)
        x = x + a
        return x + self.ff(self.ln2(x)), kc, vc


class ChordNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        blocks, cin, f = [], 1, cfg.n_bins
        for c in cfg.cnn_channels:
            blocks.append(CausalConvBlock(cin, c, centered=cfg.bidirectional))
            cin, f = c, f // 2
        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(cin * f, cfg.d_model), nn.LayerNorm(cfg.d_model), nn.Dropout(cfg.dropout))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        if cfg.sheet_delay:
            self.sheet_head = nn.Linear(cfg.d_model, cfg.n_classes)
        if cfg.key_input:
            self.key_proj = nn.Linear(12, cfg.d_model)
        if cfg.structured:
            self.root_head = nn.Linear(cfg.d_model, 13)  # 12 pitch classes + "no chord"
            self.bass_head = nn.Linear(cfg.d_model, 13)
            self.notes_head = nn.Linear(cfg.d_model, 12)  # which pitch classes sound (multi-label)
        self.register_buffer("slopes", alibi_slopes(cfg.n_heads), persistent=False)

    def attn_bias(self, T: int, device) -> torch.Tensor:
        pos = torch.arange(T, device=device)
        dist = (pos[:, None] - pos[None, :]).float()  # i - j, >= 0 for the past
        if self.cfg.bidirectional:  # offline teacher: symmetric band, distance in both directions
            bias = -self.slopes.to(device)[:, None, None] * dist.abs()
            return bias.masked_fill(dist.abs() >= self.cfg.context_frames, float("-inf"))[None]
        bias = -self.slopes.to(device)[:, None, None] * dist  # (H, T, T)
        outside = (dist < 0) | (dist >= self.cfg.context_frames)
        return bias.masked_fill(outside, float("-inf"))[None]

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        nb = self.cfg.n_bins
        h = self.cnn(x[:, None, :, :nb])  # (B, C, T, F')
        h = self.proj(h.permute(0, 2, 1, 3).reshape(B, T, -1))
        if self.cfg.key_input:  # columns after the CQT bins: key-signature vector (E20)
            h = h + self.key_proj(x[:, :, nb : nb + 12])
        return h

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        bias = self.attn_bias(x.shape[1], x.device)
        for blk in self.blocks:
            h = blk(h, bias)
        return self.ln_f(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def forward_all(self, x: torch.Tensor) -> dict:
        """Chord logits plus the structured heads (training only)."""
        h = self.features(x)
        out = {"chord": self.head(h)}
        if self.cfg.sheet_delay:
            out["sheet"] = self.sheet_head(h)
        if self.cfg.structured:
            out.update(root=self.root_head(h), bass=self.bass_head(h), notes=self.notes_head(h))
        return out

    def cache_shape(self, batch: int = 1) -> tuple[int, ...]:
        c = self.cfg
        return (c.n_layers, batch, c.n_heads, c.context_frames - 1, c.d_model // c.n_heads)


class StreamingStep(nn.Module):
    """ONNX export target. One call == one 40 ms frame.

    inputs : frames (1, RF, F)  newest RF CQT frames (RF = CNN receptive field = 13)
             k_cache, v_cache (L, 1, H, W-1, dh)
    outputs: probs (1, n_classes), new k_cache, new v_cache
             [+ sheet_probs (1, n_classes) when the model has a sheet head (E26)]
    """

    def __init__(self, net: ChordNet):
        super().__init__()
        self.net = net
        W = net.cfg.context_frames
        dist = torch.arange(W - 1, -1, -1).float()  # cache slot j is W-1-j frames old
        self.register_buffer("bias", (-net.slopes[:, None] * dist)[None, :, None, :], persistent=False)

    def forward(self, frames, k_cache, v_cache):
        h = self.net.embed(frames)[:, -1:]
        ks, vs = [], []
        for i, blk in enumerate(self.net.blocks):
            h, k, v = blk.step(h, k_cache[i], v_cache[i], self.bias)
            ks.append(k)
            vs.append(v)
        h = self.net.ln_f(h)
        probs = torch.softmax(self.net.head(h)[:, 0], dim=-1)
        if self.net.cfg.sheet_delay:
            sheet = torch.softmax(self.net.sheet_head(h)[:, 0], dim=-1)
            return probs, torch.stack(ks), torch.stack(vs), sheet
        return probs, torch.stack(ks), torch.stack(vs)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
