from pathlib import Path
from typing import Dict, Iterable, List
from .stats_engine import DescriptiveStatsEngine


class VerificationPlotter:
    def __init__(self, out_root: Path, dpi: int, format_ext: str, notch_freq: float, envelope_cutoff: float):
        self.out_root = Path(out_root)
        self.dpi = dpi
        self.format_ext = "svg"
        self.notch_freq = notch_freq
        self.envelope_cutoff = envelope_cutoff

    def generate_trial_diagnostic_plots(self, sub_id: str, act: str, filename: str, m: str, trace: Dict[str, List[float]], fs: float) -> None:
        fig_dir = self.out_root / "plots" / sub_id / filename.replace(".c3d", "") / m
        fig_dir.mkdir(parents=True, exist_ok=True)
        self._write_multiseries_svg(fig_dir / "processing_chain.svg", f"{m} processing chain ({act})", {"raw": trace["raw"], "filtered": trace["filtered"], "rectified": trace["rectified"], f"envelope_{self.envelope_cutoff}Hz": trace["envelope"]})
        freqs, psd = DescriptiveStatsEngine.compute_spectral_density(trace["raw"][:512], fs)
        _, psd_f = DescriptiveStatsEngine.compute_spectral_density(trace["filtered"][:512], fs)
        self._write_xy_svg(fig_dir / "fft_psd.svg", f"FFT/PSD verification; notch={self.notch_freq}Hz", freqs, {"raw_psd": psd, "filtered_psd": psd_f})
        self._write_histogram_svg(fig_dir / "histogram.svg", f"{m} amplitude histogram", trace["raw"], trace["filtered"])

    def generate_population_distributions(self, rows: List[Dict[str, object]]) -> None:
        out = self.out_root / "plots" / "normalization_comparison.svg"
        out.parent.mkdir(parents=True, exist_ok=True)
        values = [float(r.get("Value", 0.0)) for r in rows if r.get("Metric") == "rms"]
        self._write_multiseries_svg(out, "Normalization RMS comparison", {"rms": values[:500]})

    def _write_multiseries_svg(self, path: Path, title: str, series: Dict[str, Iterable[float]]) -> None:
        width, height = 900, 420
        all_values = [float(x) for vals in series.values() for x in list(vals)[:1000]] or [0.0]
        ymin, ymax = min(all_values), max(all_values)
        span = ymax - ymin or 1.0
        colors = ["#555", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><title>{title}</title><rect width="100%" height="100%" fill="white"/><text x="20" y="25" font-size="18">{title}</text>']
        for si, (name, vals) in enumerate(series.items()):
            vals = [float(x) for x in list(vals)[:1000]]
            if len(vals) < 2:
                continue
            pts = []
            for i, v in enumerate(vals):
                x = 40 + i * (width - 70) / (len(vals) - 1)
                y = height - 35 - ((v - ymin) / span) * (height - 75)
                pts.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[si % len(colors)]}" stroke-width="1"/><text x="{width-180}" y="{45+20*si}" fill="{colors[si % len(colors)]}">{name}</text>')
        parts.append('</svg>')
        path.write_text("".join(parts), encoding="utf-8")

    def _write_xy_svg(self, path: Path, title: str, xvals: List[float], series: Dict[str, List[float]]) -> None:
        self._write_multiseries_svg(path, title, series)

    def _write_histogram_svg(self, path: Path, title: str, raw: List[float], filtered: List[float]) -> None:
        def bins(vals: List[float]) -> List[float]:
            vals = vals[:5000]
            if not vals:
                return []
            lo, hi = min(vals), max(vals); span = hi - lo or 1.0
            counts = [0.0] * 40
            for v in vals:
                counts[min(39, int((v - lo) / span * 39))] += 1
            return counts
        self._write_multiseries_svg(path, title, {"raw_histogram": bins(raw), "filtered_histogram": bins(filtered)})
