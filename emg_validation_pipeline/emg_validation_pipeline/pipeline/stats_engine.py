import numpy as np
from scipy.signal import welch
from typing import Dict

class DescriptiveStatsEngine:
    @staticmethod
    def calculate_descriptive_metrics(sig: np.ndarray) -> Dict[str, float]:
        """Computes statistical variables to document data distributions."""
        return {
            "min": float(np.min(sig)),
            "max": float(np.max(sig)),
            "mean": float(np.mean(sig)),
            "median": float(np.median(sig)),
            "std": float(np.std(sig)),
            "variance": float(np.var(sig)),
            "rms": float(np.sqrt(np.mean(sig**2))),
            "peak": float(np.max(np.abs(sig))),
            "p95": float(np.percentile(sig, 95)),
            "p99": float(np.percentile(sig, 99)),
            "dynamic_range": float(np.max(sig) - np.min(sig))
        }

    @staticmethod
    def compute_spectral_density(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Applies Welch's method to compute power spectral densities for verification steps."""
        f, psd = welch(sig, fs=fs, nperseg=min(len(sig), 1024))
        return f, psd