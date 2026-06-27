import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict

class VerificationPlotter:
    def __init__(self, out_root: Path, dpi: int, format_ext: str):
        self.out_root = Path(out_root)
        self.dpi = dpi
        self.format_ext = format_ext

    def generate_trial_diagnostic_plots(self, sub_id: str, act: str, filename: str, m: str, trace: Dict[str, np.ndarray], fs: float) -> None:
        """Generates raw vs. processed charts on a single plot."""
        fig_dir = self.out_root / "QC" / sub_id / filename.replace(".c3d", "") / m
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Processing Chain Plot
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        t = np.arange(len(trace["raw"])) / fs
        
        axs[0].plot(t, trace["raw"], color='gray', alpha=0.7)
        axs[0].set_title(f"{m} - Raw Diagnostic Stream (DC Offset Removed)")
        
        axs[1].plot(t, trace["notch"], color='blue', alpha=0.7)
        axs[1].set_title("Filtered Stream (Bandpass + Notch Applied)")
        
        axs[2].plot(t, trace["rectified"], color='orange', alpha=0.5)
        axs[2].set_title("Full-Wave Rectified Signal")
        
        axs[3].plot(t, trace["envelope"], color='red', linewidth=2)
        axs[3].set_title("Linear Enveloping Profile (6Hz Low-pass Cutoff)")
        
        for ax in axs: ax.grid(True, linestyle=':')
        plt.xlabel("Time (seconds)")
        plt.tight_layout()
        plt.savefig(fig_dir / f"filter_comparison.{self.format_ext}", dpi=self.dpi)
        plt.close()

        # 2. Spectral Verification Plot (FFT & PSD)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Before / After FFT
        f_raw = np.fft.rfftfreq(len(trace["raw"]), 1/fs)
        fft_raw = np.abs(np.fft.rfft(trace["raw"]))
        fft_filt = np.abs(np.fft.rfft(trace["notch"]))
        
        ax1.plot(f_raw, fft_raw, label="Raw", color='gray', alpha=0.5)
        ax1.plot(f_raw, fft_filt, label="Filtered", color='green', alpha=0.8)
        ax1.set_xlim(0, 500)
        ax1.set_title("Fast Fourier Transform Frequency Analysis")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.legend()
        ax1.grid(True)
        
        # Verification Check Marker for 50Hz Noise Removal
        ax1.axvline(50, color='red', linestyle='--', alpha=0.4, label="50Hz Main Rail")

        # Probability Distribution (Histogram)
        sns.histplot(trace["raw"], kde=True, color="gray", ax=ax2, label="Raw", stat="density", alpha=0.3)
        sns.histplot(trace["notch"], kde=True, color="green", ax=ax2, label="Filtered", stat="density", alpha=0.3)
        ax2.set_title("Amplitude Distribution Curve Change")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"spectral_identity.{self.format_ext}", dpi=self.dpi)
        plt.close()

    def generate_population_distributions(self, summary_df, before_col: str, after_col: str, title: str) -> None:
        """Generates violin and box plots to check normalization behavior across subjects."""
        pop_dir = self.out_root / "QC" / "Population_Profiles"
        pop_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.violinplot(data=summary_df, x="Muscle", y=before_col, hue="Subject", ax=ax1)
        ax1.set_title(f"{title} - Raw Inter-Subject Variances")
        ax1.tick_params(axis='x', rotation=45)
        
        sns.violinplot(data=summary_df, x="Muscle", y=after_col, hue="Subject", ax=ax2)
        ax2.set_title(f"{title} - Post-Normalization Alignment")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(pop_dir / f"normalization_verification_profile.{self.format_ext}", dpi=self.dpi)
        plt.close()