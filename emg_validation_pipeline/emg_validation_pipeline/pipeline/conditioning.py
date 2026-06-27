import math
from typing import Dict, Iterable, List


class ScientificEMGConditioner:
    """Offline EMG preprocessing with auditable intermediate traces."""

    def __init__(self, fs: float, bp_low: float, bp_high: float, notch_freq: float, notch_q: float, lp_envelope: float, order: int):
        self.fs = float(fs)
        self.bp_low = float(bp_low)
        self.bp_high = float(bp_high)
        self.notch_freq = float(notch_freq)
        self.notch_q = float(notch_q)
        self.lp_envelope = float(lp_envelope)
        self.order = int(order)

    def execution_chain_trace(self, raw: Iterable[float]) -> Dict[str, List[float]]:
        values = [float(x) for x in raw]
        if not values:
            return {k: [] for k in ["raw", "bandpass", "notch", "filtered", "rectified", "envelope"]}
        mean = sum(values) / len(values)
        centered = [x - mean for x in values]
        hp = self._highpass(centered, self.bp_low)
        bandpass = self._lowpass(hp, self.bp_high)
        notch = self._notch_sine_projection(bandpass, self.notch_freq)
        rectified = [abs(x) for x in notch]
        envelope = [max(0.0, x) for x in self._lowpass(rectified, self.lp_envelope)]
        return {"raw": centered, "bandpass": bandpass, "notch": notch, "filtered": notch, "rectified": rectified, "envelope": envelope}

    def _lowpass(self, values: List[float], cutoff: float) -> List[float]:
        if not values or cutoff <= 0:
            return values[:]
        rc = 1.0 / (2.0 * math.pi * cutoff)
        dt = 1.0 / self.fs
        alpha = dt / (rc + dt)
        out = [values[0]]
        for x in values[1:]:
            out.append(out[-1] + alpha * (x - out[-1]))
        # backward pass approximates zero-phase smoothing for offline analysis
        rev = [out[-1]]
        for x in reversed(out[:-1]):
            rev.append(rev[-1] + alpha * (x - rev[-1]))
        return list(reversed(rev))

    def _highpass(self, values: List[float], cutoff: float) -> List[float]:
        low = self._lowpass(values, cutoff)
        return [x - l for x, l in zip(values, low)]

    def _notch_sine_projection(self, values: List[float], freq: float) -> List[float]:
        if not values or freq <= 0 or freq >= self.fs / 2:
            return values[:]
        sin_basis = [math.sin(2.0 * math.pi * freq * i / self.fs) for i in range(len(values))]
        cos_basis = [math.cos(2.0 * math.pi * freq * i / self.fs) for i in range(len(values))]
        denom_s = sum(x * x for x in sin_basis) or 1.0
        denom_c = sum(x * x for x in cos_basis) or 1.0
        coef_s = sum(v * b for v, b in zip(values, sin_basis)) / denom_s
        coef_c = sum(v * b for v, b in zip(values, cos_basis)) / denom_c
        return [v - coef_s * s - coef_c * c for v, s, c in zip(values, sin_basis, cos_basis)]
