from typing import Any, Dict, List
from .stats_engine import DescriptiveStatsEngine


class ConfigurableNormalizer:
    def __init__(self, strategy: str, percentile_val: float):
        self.strategy = strategy
        self.percentile_val = float(percentile_val)
        self.registry: Dict[str, float] = {}

    def build_subject_registry(self, trials: List[Dict[str, Any]]) -> None:
        for t in trials:
            if t.get("recording_type") == "continuous":
                continue
            sub = t["subject_id"]
            for muscle, env in t.get("envelopes", {}).items():
                key = f"{sub}_{muscle}"
                peak = DescriptiveStatsEngine.percentile(env, self.percentile_val)
                self.registry[key] = max(self.registry.get(key, 0.0), peak)

    def apply_normalization(self, sub_id: str, envelopes: Dict[str, List[float]]) -> Dict[str, List[float]]:
        norm_map: Dict[str, List[float]] = {}
        for m, env in envelopes.items():
            if self.strategy in {"per_trial", "percentile"}:
                denom = DescriptiveStatsEngine.percentile(env, self.percentile_val)
            elif self.strategy == "subject_maximum":
                denom = self.registry.get(f"{sub_id}_{m}", DescriptiveStatsEngine.percentile(env, self.percentile_val))
            elif self.strategy == "none":
                denom = 1.0
            elif self.strategy == "mvc":
                raise ValueError("MVC normalization requires explicit MVC reference values; use subject_maximum/percentile until mvc_dir parsing is configured.")
            else:
                raise ValueError(f"Unsupported normalization strategy: {self.strategy}")
            denom = denom if abs(denom) > 1e-12 else 1.0
            norm_map[m] = [float(x) / denom for x in env]
        return norm_map
