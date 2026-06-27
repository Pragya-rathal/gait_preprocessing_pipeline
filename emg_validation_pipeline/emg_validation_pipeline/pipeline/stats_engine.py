import math
from typing import Dict, Iterable, List, Tuple


class DescriptiveStatsEngine:
    @staticmethod
    def calculate_descriptive_metrics(sig: Iterable[float]) -> Dict[str, float]:
        values = [float(x) for x in sig]
        if not values:
            return {k: 0.0 for k in ["min", "max", "mean", "median", "std", "variance", "rms", "peak", "p95", "p99", "dynamic_range"]}
        ordered = sorted(values)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "median": DescriptiveStatsEngine.percentile(ordered, 50),
            "std": math.sqrt(variance),
            "variance": variance,
            "rms": math.sqrt(sum(x * x for x in values) / len(values)),
            "peak": max(abs(x) for x in values),
            "p95": DescriptiveStatsEngine.percentile(ordered, 95),
            "p99": DescriptiveStatsEngine.percentile(ordered, 99),
            "dynamic_range": max(values) - min(values),
        }

    @staticmethod
    def percentile(values: Iterable[float], pct: float) -> float:
        ordered = list(values)
        if not ordered:
            return 0.0
        if ordered != sorted(ordered):
            ordered = sorted(ordered)
        k = (len(ordered) - 1) * pct / 100.0
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return float(ordered[int(k)])
        return float(ordered[lo] * (hi - k) + ordered[hi] * (k - lo))

    @staticmethod
    def compute_spectral_density(sig: Iterable[float], fs: float) -> Tuple[List[float], List[float]]:
        values = [float(x) for x in sig]
        n = len(values)
        if n == 0:
            return [], []
        freqs: List[float] = []
        psd: List[float] = []
        limit = min(n // 2, 64)
        for k in range(limit + 1):
            re = 0.0
            im = 0.0
            for i, x in enumerate(values):
                angle = -2.0 * math.pi * k * i / n
                re += x * math.cos(angle)
                im += x * math.sin(angle)
            freqs.append(k * fs / n)
            psd.append((re * re + im * im) / max(n, 1))
        return freqs, psd
