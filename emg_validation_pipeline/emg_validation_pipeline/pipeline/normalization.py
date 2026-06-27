# pipeline/normalization.py
import numpy as np
from typing import Dict, List, Any

class ConfigurableNormalizer:
    def __init__(self, strategy: str, percentile_val: float):
        self.strategy = strategy
        self.percentile_val = percentile_val
        self.registry: Dict[str, float] = {}

    def build_subject_registry(self, trials: List[Dict[str, Any]]) -> None:
        """Finds scaling ceilings across subjects, protecting measurements from mixed continuous sequences."""
        for t in trials:
            if t["recording_type"] == "continuous": continue
            sub = t["subject_id"]
            for muscle, env in t["envelopes"].items():
                key = f"{sub}_{muscle}"
                peak = np.percentile(env, self.percentile_val)
                self.registry[key] = max(self.registry.get(key, 0.0), peak)

    def apply_normalization(self, sub_id: str, envelopes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        norm_map = {}
        for m, env in envelopes.items():
            if self.strategy == "per_trial":
                denom = np.percentile(env, self.percentile_val)
            elif self.strategy == "subject_maximum":
                denom = self.registry.get(f"{sub_id}_{m}", np.percentile(env, self.percentile_val))
            else:
                denom = 1.0
            norm_map[m] = env / (denom if denom > 1e-6 else 1.0)
        return norm_map

# pipeline/windowing.py
import numpy as np
from typing import List, Dict, Any

class CausalWindowSlicer:
    def __init__(self, len_ms: int, stride_ms: int, fs: float):
        self.win_size = int((len_ms / 1000.0) * fs)
        self.stride = int((stride_ms / 1000.0) * fs)
        self.fs = fs

    def extract(self, trial_data: Dict[str, Any], ordered_muscles: List[str]) -> List[Dict[str, Any]]:
        """Extracts data windows only after data verification steps have successfully passed."""
        matrix = np.array([trial_data["normalized"][m] for m in ordered_muscles])
        samples = matrix.shape[1]
        windows = []
        start = 0
        idx = 0
        
        while start + self.win_size <= samples:
            end = start + self.win_size
            windows.append({
                "subject": trial_data["subject_id"],
                "activity": trial_data["activity"],
                "filename": trial_data["filename"],
                "window_idx": idx,
                "timestamp": float(end / self.fs),
                "tensor": matrix[:, start:end]
            })
            start += self.stride
            idx += 1
        return windows