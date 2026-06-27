#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
import numpy as np

from pipeline.utils import load_config, setup_logger, BiomechanicalJustification
from pipeline.loader import C3DValidationLoader
from pipeline.conditioning import ScientificEMGConditioner
from pipeline.quality_control import SignalQualityAuditor
from pipeline.stats_engine import DescriptiveStatsEngine
from pipeline.normalization import ConfigurableNormalizer
from pipeline.windowing import CausalWindowSlicer
from pipeline.visualization import VerificationPlotter

def execute_verification_pipeline(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    out_dir = Path(cfg["data"]["output_dir"])
    logger = setup_logger(out_dir)
    
    logger.info("=================================================================")
    logger.info("   LAUNCHING BIOMECHANICAL EMG VERIFICATION & QC PIPELINE")
    logger.info("=================================================================")

    # Initializing Architecture Engines
    loader = C3DValidationLoader(cfg["data"]["raw_dir"], cfg["data"]["target_muscles"], cfg["data"]["ignored_patterns"])
    conditioner = ScientificEMGConditioner(
        fs=cfg["signal_processing"]["emg_fs"], bp_low=cfg["signal_processing"]["bandpass_low"],
        bp_high=cfg["signal_processing"]["bandpass_high"], notch_freq=cfg["signal_processing"]["notch_freq"],
        notch_q=cfg["signal_processing"]["notch_q"], lp_envelope=cfg["signal_processing"]["envelope_cutoff"],
        order=cfg["signal_processing"]["filter_order"]
    )
    auditor = SignalQualityAuditor(
        dead_thresh=cfg["qc_thresholds"]["dead_channel_ptp_v"], sat_thresh=cfg["qc_thresholds"]["saturation_v"],
        sat_pct=cfg["qc_thresholds"]["saturation_allowed_pct"], max_drift=cfg["qc_thresholds"]["max_baseline_drift_v"]
    )
    normalizer = ConfigurableNormalizer(strategy=cfg["normalization"]["strategy"], percentile_val=cfg["normalization"]["percentile_value"])
    plotter = VerificationPlotter(out_root=out_dir, dpi=cfg["plots"]["dpi"], format_ext=cfg["plots"]["format"])
    slicer = CausalWindowSlicer(len_ms=cfg["windowing"]["length_ms"], stride_ms=cfg["windowing"]["stride_ms"], fs=cfg["signal_processing"]["emg_fs"])

    c3d_files = sorted([p for p in Path(cfg["data"]["raw_dir"]).rglob("*.c3d")])
    logger.info(f"Located {len(c3d_files)} files in data storage.")

    parsed_trials = []
    global_stat_records = []
    recording_summaries = []
    qc_manifest_rows = []

    # FIRST PASS: Extract signals, run filter chains, and analyze data quality metrics
    for path in c3d_files:
        logger.info(f"Analyzing signal integrity parameters: {path.name}")
        meta = loader.parse_path_topology(path)
        trial = loader.extract_synchronized_streams(meta)
        
        # Verify synchronization alignment across hardware platforms
        duration_mismatch = abs(trial["emg_duration"] - trial["mocap_duration"])
        if duration_mismatch > 0.05:
            logger.warning(f"[Sync Mismatch Alert] Time synchronization deviation detected in {path.name}: {duration_mismatch:.3f}s difference.")

        # Filter signals and log progress metrics
        trial["traces"] = {}
        trial["envelopes"] = {}
        
        for m in cfg["data"]["target_muscles"]:
            if m in trial["emg_data_map"]:
                trace = conditioner.execution_chain_trace(trial["emg_data_map"][m])
                trial["traces"][m] = trace
                trial["envelopes"][m] = trace["envelope"]
                
                # Compute before/after stats
                stat_pre = DescriptiveStatsEngine.calculate_descriptive_metrics(trace["raw"])
                stat_post = DescriptiveStatsEngine.calculate_descriptive_metrics(trace["envelope"])
                
                for k, v in stat_pre.items():
                    global_stat_records.append({"Subject": trial["subject_id"], "Muscle": m, "File": path.name, "Stage": "Raw", "Metric": k, "Value": v})
                for k, v in stat_post.items():
                    global_stat_records.append({"Subject": trial["subject_id"], "Muscle": m, "File": path.name, "Stage": "Processed", "Metric": k, "Value": v})

        # Run quality control audits
        qc_status, qc_flags, qc_metrics = auditor.audit_telemetry(trial["emg_data_map"], trial["traces"], trial["missing_muscles"])
        trial["qc_status"] = qc_status
        
        qc_manifest_rows.append({
            "Filename": trial["filename"], "Subject": trial["subject_id"], "Activity": trial["activity"],
            "QC_Status": qc_status, "Flags": ";".join(qc_flags) if qc_flags else "NONE",
            "Missing_Muscles": ";".join(trial["missing_muscles"]) if trial["missing_muscles"] else "NONE"
        })

        recording_summaries.append({
            "Subject": trial["subject_id"], "Activity": trial["activity"], "Duration": trial["emg_duration"],
            "EMG_FS": trial["fs_emg"], "Mocap_FS": trial["fs_point"], "Muscles": len(trial["emg_data_map"]),
            "Markers": trial["marker_count"], "Force_Plates": trial["force_plate_available"], "Quality_Score": 1.0 if qc_status == "PASSED" else 0.0
        })

        if cfg["plots"]["generate_all"]:
            for m in trial["traces"].keys():
                plotter.generate_trial_diagnostic_plots(trial["subject_id"], trial["activity"], trial["filename"], m, trial["traces"][m], trial["fs_emg"])

        parsed_trials.append(trial)

    # SECOND PASS: Normalize signals across subjects and generate data windows
    normalizer.build_subject_registry(parsed_trials)
    
    final_window_list = []
    final_tensor_blocks = []

    for trial in parsed_trials:
        trial["normalized"] = normalizer.apply_normalization(trial["subject_id"], trial["envelopes"])
        
        # Slices windows only if the trial meets quality requirements
        if "FAILED" in trial["qc_status"]:
            logger.warning(f"[Pipeline Bypass] Skipping file {trial['filename']} during window creation due to failed quality metrics.")
            continue
            
        trial_windows = slicer.extract(trial, cfg["data"]["target_muscles"])
        for w in trial_windows:
            final_window_list.append(w)
            final_tensor_blocks.append(w["tensor"])

    # Save data arrays and validation tables to disk
    if final_tensor_blocks:
        np.save(out_dir / "tensors" / "validated_windows.npy", np.array(final_tensor_blocks))
        
    df_manifest = pd.DataFrame(qc_manifest_rows)
    df_manifest.to_csv(out_dir / "manifest" / "windows_manifest.csv", index=False)
    
    df_rec = pd.DataFrame(recording_summaries)
    df_rec.to_csv(out_dir / "metadata" / "recording_report.csv", index=False)
    
    df_stats = pd.DataFrame(global_stat_records)
    df_stats.to_csv(out_dir / "metadata" / "signal_statistics.csv", index=False)

    # Output Peer-Reviewed Justifications Report
    with open(out_dir / "metadata" / "literature_justifications.json", "w") as f:
        json.dump(BiomechanicalJustification.LITERATURE_MAP, f, indent=4)

    logger.info("=================================================================")
    logger.info(f"   PREPROCESSING COMPLETE: {len(final_tensor_blocks)} HIGH-QUALITY TENSORS VERIFIED")
    logger.info("=================================================================")

if __name__ == "__main__":
    execute_verification_pipeline()