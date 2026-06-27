import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List
from .utils import write_json, BiomechanicalJustification


class DatasetReportWriter:
    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)

    def write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_publication_report(self, cfg: Dict[str, Any], trials: List[Dict[str, Any]], manifest_rows: List[Dict[str, Any]]) -> None:
        payload = {
            "purpose": "Publication-quality EMG preprocessing and dataset generation for downstream NMF/muscle-synergy analysis.",
            "recordings": len(trials),
            "windows": len(manifest_rows),
            "configuration": cfg,
            "scientific_justification": BiomechanicalJustification.LITERATURE_MAP,
            "outputs": ["clean traces", "window tensors", "metadata", "QC metrics", "validation plots", "recording summaries"],
        }
        write_json(self.output_root / "reports" / "publication_dataset_report.json", payload)
