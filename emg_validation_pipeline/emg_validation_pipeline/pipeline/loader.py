import os
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

try:
    import ezc3d  # type: ignore
except ImportError as exc:
    raise ImportError("CRITICAL: 'ezc3d' is missing. Pipeline terminated. Ingestion requires raw binary parsing hooks.") from exc

class C3DValidationLoader:
    def __init__(self, root_dir: Path, target_muscles: List[str], ignored_patterns: List[str]):
        self.root_dir = Path(root_dir)
        self.target_muscles = [m.upper() for m in target_muscles]
        self.ignored_patterns = [p.upper() for p in ignored_patterns]

    def parse_path_topology(self, path: Path) -> Dict[str, Any]:
        """Extracts tracking parameters relative to the dataset folder root, avoiding stem heuristics."""
        rel = path.relative_to(self.root_dir)
        parts = rel.parts
        if len(parts) < 3:
            raise ValueError(f"Path format deviation detected. Insufficient folder depth: {path}")
        return {
            "subject_id": parts[0],
            "activity": parts[1],
            "filename": path.name,
            "recording_type": "continuous" if parts[1].lower() in ["all", "walk+squat+stair"] else "isolated",
            "absolute_path": path
        }

    def extract_synchronized_streams(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts and verifies raw telemetry streams from the C3D container."""
        reader = ezc3d.c3d(str(meta["absolute_path"]))
        
        fs_mocap = float(reader["parameters"]["POINT"]["RATE"]["value"][0])
        fs_emg = float(reader["parameters"]["ANALOG"]["RATE"]["value"][0])
        
        n_mocap_frames = reader["data"]["points"].shape[2]
        n_emg_samples = reader["data"]["analogs"].shape[2]
        
        mocap_duration = n_mocap_frames / fs_mocap
        emg_duration = n_emg_samples / fs_emg
        
        # Track channel names
        analog_labels = [l.strip().upper() for l in reader["parameters"]["ANALOG"]["LABELS"]["value"]]
        raw_analogs = reader["data"]["analogs"][0, :, :]
        
        emg_data_map = {}
        unused_channels = []
        duplicate_muscles = []
        
        for idx, lbl in enumerate(analog_labels):
            is_target = False
            for target in self.target_muscles:
                if target in lbl and not any(pat in lbl for pat in self.ignored_patterns):
                    if target in emg_data_map:
                        duplicate_muscles.append(target)
                    emg_data_map[target] = raw_analogs[idx, :]
                    is_target = True
                    break
            if not is_target:
                unused_channels.append(lbl)
                
        missing_muscles = [m for m in self.target_muscles if m not in emg_data_map]
        marker_names = [m.strip().upper() for m in reader["parameters"]["POINT"]["LABELS"]["value"]]
        
        return {
            **meta,
            "fs_emg": fs_emg,
            "fs_mocap": fs_mocap,
            "mocap_duration": mocap_duration,
            "emg_duration": emg_duration,
            "emg_data_map": emg_data_map,
            "marker_count": len(marker_names),
            "force_plate_available": "FORCE_PLATFORM" in reader["parameters"],
            "imu_available": any("IMU" in k for k in reader["parameters"].keys()),
            "missing_muscles": missing_muscles,
            "duplicate_muscles": duplicate_muscles,
            "unused_channels": unused_channels
        }