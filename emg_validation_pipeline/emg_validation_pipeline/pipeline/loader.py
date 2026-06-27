import math
from pathlib import Path
from typing import Any, Dict, List

try:
    import ezc3d  # type: ignore
except ImportError:  # pragma: no cover
    ezc3d = None


class C3DValidationLoader:
    VALID_ACTIVITIES = {"walk", "sit", "squat", "stair", "all", "walk+squat+stair"}
    CONTINUOUS_ACTIVITIES = {"all", "walk+squat+stair"}
    STATIC_ACTIVITY = "static_calibration"

    def __init__(self, root_dir: Path | str, target_muscles: List[str], ignored_patterns: List[str], allow_synthetic: bool = True, synthetic_duration_s: float = 6.0, fs: float = 2000.0, include_static_calibrations: bool = False):
        self.root_dir = Path(root_dir)
        self.target_muscles = [m.upper() for m in target_muscles]
        self.ignored_patterns = [p.upper() for p in ignored_patterns]
        self.allow_synthetic = allow_synthetic
        self.synthetic_duration_s = float(synthetic_duration_s)
        self.fs = float(fs)
        self.include_static_calibrations = include_static_calibrations
        self.static_skipped_count = 0
        self.static_included_count = 0
        self.static_recordings: List[Path] = []

    def discover_files(self) -> List[Path]:
        files = sorted(self.root_dir.rglob("*.c3d")) if self.root_dir.exists() else []
        if files:
            processing_files: List[Path] = []
            self.static_recordings = []
            self.static_skipped_count = 0
            self.static_included_count = 0
            for path in files:
                if self.is_static_calibration(path):
                    self.static_recordings.append(path)
                    if self.include_static_calibrations:
                        self.static_included_count += 1
                        processing_files.append(path)
                    else:
                        self.static_skipped_count += 1
                else:
                    processing_files.append(path)
            return processing_files
        if not self.allow_synthetic:
            return []
        return [Path(f"SYNTHETIC_SUBJECT/{a}/synthetic_{a}.c3d") for a in ["walk", "sit", "squat", "stair", "all"]]

    def is_static_calibration(self, path: Path) -> bool:
        try:
            parts = path.relative_to(self.root_dir).parts
        except ValueError:
            parts = path.parts
        return len(parts) == 2 and Path(parts[-1]).suffix.lower() == ".c3d"

    def parse_path_topology(self, path: Path) -> Dict[str, Any]:
        try:
            rel = path.relative_to(self.root_dir)
            parts = rel.parts
        except ValueError:
            parts = path.parts
        if len(parts) == 2:
            return {
                "subject_id": parts[0],
                "activity": self.STATIC_ACTIVITY,
                "recording": Path(path).stem,
                "filename": Path(path).name,
                "recording_type": "static_calibration",
                "absolute_path": path,
                "synthetic": not Path(path).exists(),
                "is_static_calibration": True,
            }
        if len(parts) < 3:
            return {
                "subject_id": parts[0] if parts else "UNKNOWN_SUBJECT",
                "activity": self.STATIC_ACTIVITY,
                "recording": Path(path).stem,
                "filename": Path(path).name,
                "recording_type": "static_calibration",
                "absolute_path": path,
                "synthetic": not Path(path).exists(),
                "is_static_calibration": True,
            }
        activity = parts[-2].lower()
        if activity not in self.VALID_ACTIVITIES:
            raise ValueError(f"Unsupported activity folder '{activity}' in {path}. Expected one of {sorted(self.VALID_ACTIVITIES)} or a static calibration C3D directly under a subject directory.")
        return {
            "subject_id": parts[-3],
            "activity": activity,
            "recording": Path(path).stem,
            "filename": Path(path).name,
            "recording_type": "continuous" if activity in self.CONTINUOUS_ACTIVITIES else "isolated",
            "absolute_path": path,
            "synthetic": not Path(path).exists(),
            "is_static_calibration": False,
        }

    def extract_synchronized_streams(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        if meta.get("synthetic") or ezc3d is None:
            return self._synthetic_trial(meta)
        reader = ezc3d.c3d(str(meta["absolute_path"]))
        fs_mocap = float(reader["parameters"]["POINT"]["RATE"]["value"][0])
        fs_emg = float(reader["parameters"]["ANALOG"]["RATE"]["value"][0])
        n_mocap_frames = reader["data"]["points"].shape[2]
        raw_analogs = reader["data"]["analogs"][0, :, :]
        n_emg_samples = raw_analogs.shape[1]
        analog_labels = [str(l).strip().upper() for l in reader["parameters"]["ANALOG"]["LABELS"]["value"]]
        emg_data_map: Dict[str, List[float]] = {}
        unused_channels: List[str] = []
        duplicate_muscles: List[str] = []
        for idx, lbl in enumerate(analog_labels):
            matched = [t for t in self.target_muscles if self._label_matches(lbl, t)]
            if matched:
                target = matched[0]
                if target in emg_data_map:
                    duplicate_muscles.append(target)
                emg_data_map[target] = [float(x) for x in raw_analogs[idx, :]]
            else:
                unused_channels.append(lbl)
        marker_names = [str(m).strip().upper() for m in reader["parameters"].get("POINT", {}).get("LABELS", {}).get("value", [])]
        return {**meta, "fs_emg": fs_emg, "fs_mocap": fs_mocap, "mocap_duration": n_mocap_frames / fs_mocap, "emg_duration": n_emg_samples / fs_emg, "emg_data_map": emg_data_map, "marker_count": len(marker_names), "force_plate_available": "FORCE_PLATFORM" in reader["parameters"], "imu_available": any("IMU" in k.upper() for k in reader["parameters"].keys()), "missing_muscles": [m for m in self.target_muscles if m not in emg_data_map], "duplicate_muscles": duplicate_muscles, "unused_channels": unused_channels}

    def _label_matches(self, label: str, target: str) -> bool:
        if any(pat in label for pat in self.ignored_patterns):
            return False
        normalized = label.replace("-", "_").replace(" ", "_").upper()
        return normalized == target or normalized.endswith("_" + target) or target in normalized.split("_")

    def _synthetic_trial(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        fs_emg = self.fs
        n = int(self.synthetic_duration_s * fs_emg)
        activity = meta["activity"]
        emg_data_map: Dict[str, List[float]] = {}
        for mi, muscle in enumerate(self.target_muscles):
            data: List[float] = []
            for i in range(n):
                t = i / fs_emg
                if activity in self.CONTINUOUS_ACTIVITIES:
                    phase = int((t / self.synthetic_duration_s) * 3)
                    amp = [0.0008, 0.0013, 0.0010][min(phase, 2)]
                else:
                    amp = {"walk": 0.0012, "sit": 0.00045, "squat": 0.0015, "stair": 0.0018}.get(activity, 0.001)
                carrier = math.sin(2 * math.pi * (70 + mi * 3) * t) + 0.35 * math.sin(2 * math.pi * 110 * t)
                envelope = 0.5 + 0.5 * math.sin(2 * math.pi * (0.8 + mi * 0.02) * t)
                data.append(amp * envelope * carrier + 0.00002 * math.sin(2 * math.pi * 60 * t))
            emg_data_map[muscle] = data
        return {**meta, "fs_emg": fs_emg, "fs_mocap": 100.0, "mocap_duration": self.synthetic_duration_s, "emg_duration": self.synthetic_duration_s, "emg_data_map": emg_data_map, "marker_count": 0, "force_plate_available": False, "imu_available": False, "missing_muscles": [], "duplicate_muscles": [], "unused_channels": []}
