from typing import Any, Dict, List, Tuple
from .stats_engine import DescriptiveStatsEngine


class SignalQualityAuditor:
    def __init__(self, dead_thresh: float, sat_thresh: float, sat_pct: float, max_drift: float, abnormal_rms_max: float = 0.05, sync_tolerance_s: float = 0.05):
        self.dead_thresh = float(dead_thresh)
        self.sat_thresh = float(sat_thresh)
        self.sat_pct = float(sat_pct)
        self.max_drift = float(max_drift)
        self.abnormal_rms_max = float(abnormal_rms_max)
        self.sync_tolerance_s = float(sync_tolerance_s)

    def audit_telemetry(self, trial: Dict[str, Any], trace_map: Dict[str, Dict[str, List[float]]]) -> Tuple[str, List[str], Dict[str, Any], float]:
        flags: List[str] = []
        metrics: Dict[str, Any] = {"channels": {}, "sync_mismatch_s": abs(trial["emg_duration"] - trial["mocap_duration"])}
        if trial.get("missing_muscles"):
            flags.append(f"MISSING_CHANNELS_{len(trial['missing_muscles'])}")
        if trial.get("duplicate_muscles"):
            flags.append(f"DUPLICATE_CHANNELS_{len(trial['duplicate_muscles'])}")
        if metrics["sync_mismatch_s"] > self.sync_tolerance_s:
            flags.append("BAD_SYNCHRONIZATION")
        for m, raw in trial["emg_data_map"].items():
            stats = DescriptiveStatsEngine.calculate_descriptive_metrics(raw)
            ptp = stats["dynamic_range"]
            sat_ratio = sum(1 for x in raw if abs(x) >= self.sat_thresh) / max(len(raw), 1)
            chunks = _chunks(raw, 10)
            drift = max([sum(c) / len(c) for c in chunks]) - min([sum(c) / len(c) for c in chunks]) if chunks else 0.0
            if ptp < self.dead_thresh:
                flags.append(f"{m}_DEAD_CHANNEL")
            if sat_ratio > self.sat_pct:
                flags.append(f"{m}_CLIPPED_CHANNEL")
            if drift > self.max_drift:
                flags.append(f"{m}_BASELINE_DRIFT")
            if stats["rms"] > self.abnormal_rms_max:
                flags.append(f"{m}_ABNORMAL_AMPLITUDE")
            metrics["channels"][m] = {**stats, "ptp_voltage": ptp, "saturation_pct": sat_ratio * 100.0, "baseline_drift": drift, "envelope_rms": DescriptiveStatsEngine.calculate_descriptive_metrics(trace_map.get(m, {}).get("envelope", []))["rms"]}
        critical = [f for f in flags if any(key in f for key in ["DEAD", "CLIPPED", "MISSING", "CORRUPTED", "BAD_SYNCHRONIZATION", "ABNORMAL"])]
        score = max(0.0, 1.0 - 0.15 * len(flags) - 0.25 * len(critical))
        status = "PASSED" if not critical else "FAILED_" + "_".join(critical[:3])
        return status, flags, metrics, score


def _chunks(values: List[float], count: int) -> List[List[float]]:
    if not values:
        return []
    size = max(1, len(values) // count)
    return [values[i:i + size] for i in range(0, len(values), size)]
