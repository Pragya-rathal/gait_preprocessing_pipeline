from typing import Any, Dict, List


class ResearchWindowSlicer:
    def __init__(self, len_ms: int, stride_ms: int, fs: float, transition_margin_ms: int, future_horizon_ms: int):
        self.win_size = max(1, int((len_ms / 1000.0) * fs))
        self.stride = max(1, int((stride_ms / 1000.0) * fs))
        self.fs = float(fs)
        self.window_duration_s = self.win_size / self.fs
        self.overlap = 1.0 - (self.stride / self.win_size)
        self.transition_margin_s = transition_margin_ms / 1000.0
        self.future_horizon_s = future_horizon_ms / 1000.0

    def extract(self, trial_data: Dict[str, Any], ordered_muscles: List[str]) -> List[Dict[str, Any]]:
        missing = [m for m in ordered_muscles if m not in trial_data["normalized"]]
        if missing:
            raise ValueError(f"Cannot window trial with missing normalized muscles: {missing}")
        matrix = [trial_data["normalized"][m] for m in ordered_muscles]
        samples = min(len(row) for row in matrix) if matrix else 0
        transitions = self._transition_schedule(trial_data["activity"], trial_data["emg_duration"])
        windows: List[Dict[str, Any]] = []
        start = 0
        idx = 0
        while start + self.win_size <= samples:
            end = start + self.win_size
            start_time = start / self.fs
            end_time = end / self.fs
            timestamp = end_time
            current = self._label_at(trial_data["activity"], start_time, transitions)
            future = self._label_at(trial_data["activity"], timestamp + self.future_horizon_s, transitions)
            nearest = self._nearest_transition(timestamp, transitions)
            transition_flag = current != future or (nearest is not None and abs(nearest["time"] - timestamp) <= self.transition_margin_s)
            category = self._category(timestamp, nearest, transition_flag, trial_data["recording_type"])
            windows.append({
                "subject": trial_data["subject_id"],
                "recording": trial_data["recording"],
                "activity": trial_data["activity"],
                "recording_type": trial_data["recording_type"],
                "window_idx": idx,
                "start_time": start_time,
                "end_time": end_time,
                "timestamp": timestamp,
                "tensor": [row[start:end] for row in matrix],
                "emg_sampling_rate": self.fs,
                "normalization_method": trial_data["normalization_method"],
                "qc_status": trial_data["qc_status"],
                "qc_score": trial_data["qc_score"],
                "qc_metrics": trial_data["qc_metrics"],
                "missing_muscles": trial_data.get("missing_muscles", []),
                "muscles_present": list(trial_data["normalized"].keys()),
                "window_duration": self.window_duration_s,
                "overlap": self.overlap,
                "current_activity": current,
                "future_activity": future,
                "transition_flag": transition_flag,
                "transition_type": nearest["type"] if transition_flag and nearest else (f"{current}_to_{future}" if current != future else "none"),
                "window_position": self._position(start_time, end_time, trial_data["emg_duration"]),
                "time_to_transition": None if nearest is None else nearest["time"] - timestamp,
                "window_category": category,
            })
            start += self.stride
            idx += 1
        return windows

    def _transition_schedule(self, activity: str, duration: float) -> List[Dict[str, Any]]:
        if activity == "all":
            labels = ["walk", "sit", "squat", "stair"]
        elif activity == "walk+squat+stair":
            labels = ["walk", "squat", "stair"]
        else:
            return []
        seg = duration / len(labels)
        return [{"time": seg * i, "from": labels[i - 1], "to": labels[i], "type": f"{labels[i - 1]}_to_{labels[i]}"} for i in range(1, len(labels))]

    def _label_at(self, activity: str, time_s: float, transitions: List[Dict[str, Any]]) -> str:
        if not transitions:
            return activity
        label = transitions[0]["from"]
        for tr in transitions:
            if time_s >= tr["time"]:
                label = tr["to"]
        return label

    def _nearest_transition(self, timestamp: float, transitions: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not transitions:
            return None
        return min(transitions, key=lambda tr: abs(tr["time"] - timestamp))

    def _category(self, timestamp: float, nearest: Dict[str, Any] | None, flag: bool, rec_type: str) -> str:
        if nearest is None:
            return "steady-state" if rec_type == "isolated" else "background"
        delta = timestamp - nearest["time"]
        if abs(delta) <= self.transition_margin_s:
            return "transition"
        if -2 * self.transition_margin_s <= delta < -self.transition_margin_s:
            return "pre-transition"
        if self.transition_margin_s < delta <= 2 * self.transition_margin_s:
            return "post-transition"
        return "background" if rec_type == "continuous" else "steady-state"

    def _position(self, start: float, end: float, duration: float) -> str:
        midpoint = (start + end) / 2.0
        ratio = midpoint / max(duration, 1e-9)
        if ratio < 0.33:
            return "early"
        if ratio < 0.66:
            return "middle"
        return "late"

CausalWindowSlicer = ResearchWindowSlicer
