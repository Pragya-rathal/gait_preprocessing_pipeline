#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, List
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.utils import load_config, setup_logger, ensure_output_tree, write_json
from pipeline.loader import C3DValidationLoader
from pipeline.conditioning import ScientificEMGConditioner
from pipeline.quality_control import SignalQualityAuditor
from pipeline.stats_engine import DescriptiveStatsEngine
from pipeline.normalization import ConfigurableNormalizer
from pipeline.windowing import ResearchWindowSlicer
from pipeline.visualization import VerificationPlotter
from pipeline.reports import DatasetReportWriter


def execute_verification_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = Path(cfg["data"]["output_dir"])
    ensure_output_tree(out_dir)
    logger = setup_logger(out_dir)
    reporter = DatasetReportWriter(out_dir)
    logger.info("Launching EMG preprocessing and dataset generation pipeline")

    loader = C3DValidationLoader(cfg["data"]["raw_dir"], cfg["data"]["target_muscles"], cfg["data"]["ignored_patterns"], cfg["data"].get("allow_synthetic_when_empty", True), cfg["data"].get("synthetic_duration_s", 6.0), cfg["signal_processing"]["emg_fs"])
    auditor = SignalQualityAuditor(cfg["qc_thresholds"]["dead_channel_ptp_v"], cfg["qc_thresholds"]["saturation_v"], cfg["qc_thresholds"]["saturation_allowed_pct"], cfg["qc_thresholds"]["max_baseline_drift_v"], cfg["qc_thresholds"].get("abnormal_rms_max_v", 0.05), cfg["qc_thresholds"].get("sync_tolerance_s", 0.05))
    normalizer = ConfigurableNormalizer(cfg["normalization"]["strategy"], cfg["normalization"]["percentile_value"])
    plotter = VerificationPlotter(out_dir, cfg["plots"]["dpi"], cfg["plots"].get("format", "svg"), cfg["signal_processing"]["notch_freq"], cfg["signal_processing"]["envelope_cutoff"])

    parsed_trials: List[Dict[str, Any]] = []
    stat_rows: List[Dict[str, Any]] = []
    recording_rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []

    files = loader.discover_files()
    logger.info("Located %s C3D-compatible recording(s)", len(files))
    for path in files:
        meta = loader.parse_path_topology(path)
        trial = loader.extract_synchronized_streams(meta)
        if abs(trial["fs_emg"] - cfg["signal_processing"]["emg_fs"]) > 1e-6:
            logger.warning("%s EMG rate %.3f differs from configured %.3f; using file rate", trial["filename"], trial["fs_emg"], cfg["signal_processing"]["emg_fs"])
        conditioner = ScientificEMGConditioner(trial["fs_emg"], cfg["signal_processing"]["bandpass_low"], cfg["signal_processing"]["bandpass_high"], cfg["signal_processing"]["notch_freq"], cfg["signal_processing"]["notch_q"], cfg["signal_processing"]["envelope_cutoff"], cfg["signal_processing"]["filter_order"])
        trial["traces"] = {}
        trial["envelopes"] = {}
        for muscle in cfg["data"]["target_muscles"]:
            muscle = muscle.upper()
            if muscle not in trial["emg_data_map"]:
                continue
            trace = conditioner.execution_chain_trace(trial["emg_data_map"][muscle])
            trial["traces"][muscle] = trace
            trial["envelopes"][muscle] = trace["envelope"]
            for stage in ["raw", "filtered", "rectified", "envelope"]:
                for metric, value in DescriptiveStatsEngine.calculate_descriptive_metrics(trace[stage]).items():
                    stat_rows.append({"Subject": trial["subject_id"], "Recording": trial["recording"], "Activity": trial["activity"], "Muscle": muscle, "Stage": stage, "Metric": metric, "Value": value})
            if cfg["plots"].get("generate_all", True):
                plotter.generate_trial_diagnostic_plots(trial["subject_id"], trial["activity"], trial["filename"], muscle, trace, trial["fs_emg"])
        qc_status, qc_flags, qc_metrics, qc_score = auditor.audit_telemetry(trial, trial["traces"])
        trial.update({"qc_status": qc_status, "qc_flags": qc_flags, "qc_metrics": qc_metrics, "qc_score": qc_score})
        qc_rows.append({"Filename": trial["filename"], "Subject": trial["subject_id"], "Recording": trial["recording"], "Activity": trial["activity"], "Recording_Type": trial["recording_type"], "QC_Status": qc_status, "QC_Score": qc_score, "Flags": ";".join(qc_flags) if qc_flags else "NONE", "Missing_Muscles": ";".join(trial["missing_muscles"]) if trial["missing_muscles"] else "NONE"})
        recording_rows.append({"Subject": trial["subject_id"], "Recording": trial["recording"], "Activity": trial["activity"], "Recording_Type": trial["recording_type"], "Duration": trial["emg_duration"], "EMG_FS": trial["fs_emg"], "Mocap_FS": trial["fs_mocap"], "Muscles": len(trial["emg_data_map"]), "Markers": trial["marker_count"], "Force_Plates": trial["force_plate_available"], "Quality_Score": qc_score, "QC_Status": qc_status})
        parsed_trials.append(trial)

    normalizer.build_subject_registry(parsed_trials)
    manifest_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    tensor_index = 0
    for trial in parsed_trials:
        trial["normalization_method"] = cfg["normalization"]["strategy"]
        trial["normalized"] = normalizer.apply_normalization(trial["subject_id"], trial["envelopes"])
        if trial["qc_status"].startswith("FAILED") and not cfg["qc_thresholds"].get("export_failed_windows", False):
            logger.warning("Skipping failed recording %s: %s", trial["filename"], trial["qc_status"])
            continue
        slicer = ResearchWindowSlicer(cfg["windowing"]["length_ms"], cfg["windowing"]["stride_ms"], trial["fs_emg"], cfg["windowing"].get("transition_margin_ms", 500), cfg["windowing"].get("anticipatory_horizon_ms", 300))
        for window in slicer.extract(trial, [m.upper() for m in cfg["data"]["target_muscles"]]):
            tensor_name = f"tensor_{tensor_index:08d}.json"
            tensor_path = out_dir / "tensors" / tensor_name
            write_json(tensor_path, window["tensor"])
            row = {"Tensor filename": tensor_name, "Tensor Path": str(tensor_path), "Tensor Shape": f"{len(window['tensor'])}x{len(window['tensor'][0]) if window['tensor'] else 0}", "Subject": window["subject"], "Recording": window["recording"], "Activity": window["activity"], "Recording Type": window["recording_type"], "Window Index": window["window_idx"], "Start Time": window["start_time"], "End Time": window["end_time"], "Timestamp": window["timestamp"], "EMG Sampling Rate": window["emg_sampling_rate"], "Normalization Method": window["normalization_method"], "QC Status": window["qc_status"], "QC Score": window["qc_score"], "Missing Muscles": ";".join(window["missing_muscles"]) if window["missing_muscles"] else "NONE", "Muscles Present": ";".join(window["muscles_present"]), "Window Duration": window["window_duration"], "Overlap": window["overlap"], "Current Activity": window["current_activity"], "Future Activity": window["future_activity"], "Current Label": window["current_activity"], "Future Label": window["future_activity"], "Transition Flag": window["transition_flag"], "Transition Type": window["transition_type"], "Window Position": window["window_position"], "Time To Transition": window["time_to_transition"], "Window Category": window["window_category"]}
            manifest_rows.append(row)
            metadata_rows.append({**row, "QC Metrics JSON": window["qc_metrics"]})
            tensor_index += 1

    reporter.write_csv(out_dir / "manifest" / "dataset_manifest.csv", manifest_rows)
    reporter.write_csv(out_dir / "metadata" / "window_metadata.csv", metadata_rows)
    reporter.write_csv(out_dir / "metadata" / "recording_report.csv", recording_rows)
    reporter.write_csv(out_dir / "metadata" / "signal_statistics.csv", stat_rows)
    reporter.write_csv(out_dir / "qc" / "qc_manifest.csv", qc_rows)
    write_json(out_dir / "qc" / "qc_metrics.json", [{"trial": t["filename"], "metrics": t["qc_metrics"]} for t in parsed_trials])
    reporter.write_publication_report(cfg, parsed_trials, manifest_rows)
    plotter.generate_population_distributions(stat_rows)
    logger.info("Preprocessing complete: %s tensors generated in %s", len(manifest_rows), out_dir)
    return {"recordings": len(parsed_trials), "windows": len(manifest_rows), "output_dir": str(out_dir)}


if __name__ == "__main__":
    execute_verification_pipeline()
