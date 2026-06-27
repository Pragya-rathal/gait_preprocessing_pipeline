import numpy as np
from typing import Dict, List, Tuple, Any

class SignalQualityAuditor:
    def __init__(self, dead_thresh: float, sat_thresh: float, sat_pct: float, max_drift: float):
        self.dead_thresh = dead_thresh
        self.sat_thresh = sat_thresh
        self.sat_pct = sat_pct
        self.max_drift = max_drift

    def audit_telemetry(self, raw_map: Dict[str, np.ndarray], trace_map: Dict[str, Dict[str, np.ndarray]], missing_m: List[str]) -> Tuple[str, List[str], Dict[str, Any]]:
        """Audits signal channels against biometric limits to flag noise and electrode dropouts."""
        flags = []
        metrics = {}
        
        if missing_m:
            flags.append(f"MISSING_CHANNELS_{len(missing_m)}")
            
        for m, raw in raw_map.items():
            ptp = np.ptp(raw)
            rms = np.sqrt(np.mean(raw**2))
            
            # Dead/disconnected channel test
            if ptp < self.dead_thresh:
                flags.append(f"{m}_DEAD_CHANNEL")
                
            # Clipping/Saturation test
            sat_samples = np.sum(np.abs(raw) >= self.sat_thresh)
            sat_ratio = sat_samples / len(raw)
            if sat_ratio > self.sat_pct:
                flags.append(f"{m}_SENSOR_SATURATION")
                
            # Baseline offset/drift test
            filtered_env = trace_map[m]["envelope"]
            chunks = np.array_split(raw, 10)
            means = [np.mean(c) for c in chunks]
            drift = np.ptp(means)
            if drift > self.max_drift:
                flags.append(f"{m}_BASELINE_DRIFT")
                
            # Track verified metrics
            metrics[m] = {
                "ptp_voltage": float(ptp),
                "saturation_pct": float(sat_ratio * 100.0),
                "baseline_drift": float(drift),
                "rms_amplitude": float(rms)
            }
            
        status = "PASSED" if not flags else f"FAILED_{'_'.join(flags[:3])}"
        return status, flags, metrics