import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

class ScientificEMGConditioner:
    """Applies zero-phase digital filtering chains to isolate the targeted biological signals."""
    def __init__(self, fs: float, bp_low: float, bp_high: float, notch_freq: float, notch_q: float, lp_envelope: float, order: int):
        self.fs = fs
        nyq = 0.5 * fs
        self.bp_sos = butter(order, [bp_low / nyq, bp_high / nyq], btype="bandpass", output="sos")
        self.notch_b, self.notch_a = iirnotch(notch_freq / nyq, notch_q)
        self.lp_sos = butter(order, lp_envelope / nyq, btype="low", output="sos")

    def execution_chain_trace(self, raw: np.ndarray) -> dict[str, np.ndarray]:
        """Processes signals through intermediate steps, maintaining an exact audit log of modifications."""
        trace = {"raw": raw.astype(np.float64) - np.mean(raw)} # Remove baseline DC offset immediately
        trace["bandpass"] = sosfiltfilt(self.bp_sos, trace["raw"])
        trace["notch"] = filtfilt(self.notch_b, self.notch_a, trace["bandpass"])
        trace["rectified"] = np.abs(trace["notch"])
        trace["envelope"] = sosfiltfilt(self.lp_sos, trace["rectified"])
        # Enforce non-negativity constraint
        trace["envelope"] = np.maximum(trace["envelope"], 0.0)
        return trace