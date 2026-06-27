# gait_preprocessing_pipeline

Publication-oriented EMG preprocessing and dataset generation pipeline for downstream NMF and muscle-synergy analysis.

Run from the repository root:

```bash
python main.py
```

The pipeline expects C3D recordings organized as `Subject/activity/file.c3d`, where activity is one of `walk`, `sit`, `squat`, `stair`, `all`, or `walk+squat+stair`. Static calibration recordings placed directly under a subject folder, such as `Subject/Static FB Anterior - IOR 1.c3d`, are detected and skipped by default; set `data.include_static_calibrations: true` to include them. If no C3D files are present, it creates a deterministic synthetic smoke-test dataset so the repository remains runnable without manual intervention.

Outputs are written to `processed_dataset/`:

- `tensors/` per-window tensors
- `manifest/` dataset manifest
- `metadata/` recording, window, and signal statistics
- `reports/` publication dataset report
- `plots/` scientific validation plots
- `qc/` quality-control manifests and metrics
