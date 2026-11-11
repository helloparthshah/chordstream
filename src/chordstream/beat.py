"""Causal beat tracker on the chord model's own CQT frames (no extra audio processing).

Per 40 ms frame it reports, using only past audio:
  phase      frames since the last beat (fractional), in [0, period)
  period     beat period in frames (fractional), from the tempo estimate
  confidence 0..1 - how periodic the recent onsets are; ~0 in intros, rubato, silence
  beat       True on the frame where a beat is predicted to fall

Upcoming beats are *extrapolated* from the tempo and phase of past audio. That is a
prediction, not lookahead, so the zero-lookahead guarantee of the system still holds.

Pipeline: spectral-flux onset envelope -> autocorrelation tempo with a log-Gaussian prior
(suppresses half/double-tempo errors) -> comb-filter phase over the last few beats ->
phase tracked with a small correction gain, so it follows the beat without jitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FPS = 25.0  # 40 ms frames
HIST = 256  # onset history (~10 s)
BPM_GRID = np.arange(55.0, 200.0, 0.5)


@dataclass
class BeatState:
    phase: float
    period: float
    confidence: float
    beat: bool

    @property
    def bpm(self) -> float:
        return 60.0 * FPS / self.period if self.period > 0 else 0.0


class BeatTracker:
    def __init__(self, fps: float = FPS, prior_bpm: float = 110.0, prior_octaves: float = 0.9,
                 tempo_every: int = 10, phase_gain: float = 0.15):
        self.fps = fps
        self.lags = 60.0 * fps / BPM_GRID  # candidate periods (frames)
        self.prior = np.exp(-0.5 * (np.log2(BPM_GRID / prior_bpm) / prior_octaves) ** 2)
        self.tempo_every, self.gain = tempo_every, phase_gain
        self.reset()

    def reset(self) -> None:
        self.prev = None
        self.onset = np.zeros(HIST, np.float32)
        self.flux_mean = 0.0
        self.n = 0
        self.period = 60.0 * self.fps / 110.0
        self.tempo_conf = 0.0
        self.phase = 0.0
        self.conf = 0.0

    # -------------------------------------------------------------- onset envelope
    def _onset(self, frame: np.ndarray) -> float:
        if self.prev is None:
            self.prev = frame
            return 0.0
        flux = float(np.maximum(frame - self.prev, 0.0).sum())
        self.prev = frame
        self.flux_mean = 0.97 * self.flux_mean + 0.03 * flux  # slow baseline (~1.3 s)
        return max(0.0, flux - self.flux_mean)

    # -------------------------------------------------------------- tempo
    def _update_tempo(self) -> None:
        o = self.onset[-min(self.n, HIST):]
        if len(o) < 4 * FPS:
            return
        x = o - o.mean()
        e = float((x * x).sum())
        if e <= 1e-9:
            self.tempo_conf = 0.0
            return
        nfft = 1 << int(np.ceil(np.log2(2 * len(x))))
        f = np.fft.rfft(x, nfft)
        acf = np.fft.irfft(f * np.conj(f), nfft)[: len(x)] / e
        lagx = np.arange(len(acf))

        def at(l):
            return np.interp(l, lagx, acf, right=0.0)

        # Metrical comb: a true period also correlates at 2x and 4x (and 0.5x for subdivisions).
        score = at(self.lags) + 0.5 * at(2 * self.lags) + 0.25 * at(4 * self.lags) + 0.25 * at(self.lags / 2)
        score = np.maximum(score, 0) * self.prior
        i = int(score.argmax())
        cand = self.lags[i]
        # Hysteresis: keep the current tempo unless the new one is clearly better.
        cur = int(np.argmin(np.abs(self.lags - self.period)))
        if abs(cand - self.period) > 0.5 and score[i] < 1.15 * score[cur]:
            cand, i = self.lags[cur], cur
        self.period = float(cand)
        self.tempo_conf = float(np.clip(at(cand), 0.0, 1.0))

    # -------------------------------------------------------------- phase
    def _measure_phase(self) -> tuple[float, float]:
        P = self.period
        o = self.onset
        T = len(o) - 1
        phis = np.arange(0.0, P, 0.25)
        k = np.arange(6)
        w = 0.8 ** k
        pos = T - phis[:, None] - k[None, :] * P  # (n_phase, n_beats)
        vals = np.interp(pos, np.arange(len(o)), o, left=0.0)
        comb = (vals * w).sum(1)
        j = int(comb.argmax())
        peak = (comb[j] - comb.mean()) / (comb.std() + 1e-6)
        return float(phis[j]), float(peak)

    def step(self, frame: np.ndarray) -> BeatState:
        self.onset[:-1] = self.onset[1:]
        self.onset[-1] = self._onset(frame)
        self.n += 1
        if self.n % self.tempo_every == 0:
            self._update_tempo()

        prev_phase = self.phase
        pred = (self.phase + 1.0) % self.period
        if self.n >= 3 * FPS:
            meas, peak = self._measure_phase()
            d = (meas - pred + self.period / 2) % self.period - self.period / 2  # circular difference
            self.phase = (pred + self.gain * d) % self.period
            comb_conf = float(np.clip((peak - 1.0) / 2.0, 0.0, 1.0))
            target = comb_conf * float(np.clip(self.tempo_conf * 2.5, 0.0, 1.0))
        else:
            self.phase, target = pred, 0.0
        self.conf = 0.95 * self.conf + 0.05 * target
        beat = self.phase < prev_phase  # wrapped: a beat boundary fell in this frame
        return BeatState(self.phase, self.period, self.conf, beat)

    def run(self, frames: np.ndarray) -> list[BeatState]:
        self.reset()
        return [self.step(f) for f in frames]


def change_scale(state: BeatState, lag: float, window: float, suppress: float) -> float:
    """Multiplier on the HMM's chord-change probability for this frame.

    Near a (lag-shifted) beat the change rate is boosted, between beats it is suppressed;
    the boost is chosen so the average change rate is unchanged. Low confidence -> 1.0.
    """
    P = state.period
    d = (state.phase - lag) % P
    dist = min(d, P - d)
    frac = min(1.0, (2 * window + 1) / P)
    boost = (1.0 - (1.0 - frac) * suppress) / frac
    g = boost if dist <= window else suppress
    return (1.0 - state.confidence) + state.confidence * g
