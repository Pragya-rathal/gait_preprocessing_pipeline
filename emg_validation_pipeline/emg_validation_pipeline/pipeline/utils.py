import json
import logging
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load and validate the pipeline configuration."""
    path = Path(config_path)
    if not path.exists():
        package_default = Path(__file__).resolve().parents[1] / "config.yaml"
        if package_default.exists():
            path = package_default
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = _minimal_yaml(text)
    required = ["data", "signal_processing", "qc_thresholds", "normalization", "windowing", "labels", "plots"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Configuration is missing required sections: {missing}")
    return cfg


def _minimal_yaml(text: str) -> Dict[str, Any]:
    """Small YAML subset parser for this repository's config when PyYAML is unavailable."""
    result: Dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, result)]
    last_key_at_indent: dict[int, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = raw.split("#", 1)[0].rstrip()
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            parent.append(_parse_scalar(stripped[2:].strip()))
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            container: Any = [] if _next_nonempty_is_list(text, raw) else {}
            parent[key] = container
            stack.append((indent, container))
            last_key_at_indent[indent] = key
        else:
            parent[key] = _parse_scalar(value)
    return result


def _next_nonempty_is_list(text: str, current: str) -> bool:
    lines = text.splitlines()
    try:
        idx = lines.index(current)
    except ValueError:
        return False
    base = len(current) - len(current.lstrip(" "))
    for nxt in lines[idx + 1:]:
        if not nxt.strip() or nxt.lstrip().startswith("#"):
            continue
        ind = len(nxt) - len(nxt.lstrip(" "))
        return ind > base and nxt.strip().startswith("- ")
    return False


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(v.strip()) for v in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("EMGValidationPipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    f_handler = logging.FileHandler(output_dir / "preprocessing_audit.log")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)
    return logger


def ensure_output_tree(output_dir: Path) -> Dict[str, Path]:
    dirs = {name: output_dir / name for name in ["tensors", "metadata", "manifest", "reports", "plots", "qc"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class BiomechanicalJustification:
    LITERATURE_MAP = {
        "bandpass": "Offline surface EMG commonly uses high-pass filtering near 20 Hz and low-pass filtering near 400-500 Hz to reduce motion artifacts while preserving myoelectric content.",
        "notch": "Power-line attenuation should match the local acquisition environment (50 or 60 Hz) and be documented in the generated report.",
        "rectification": "Full-wave rectification is a standard step before envelope extraction for amplitude-based EMG analysis.",
        "envelope": "Low-pass envelope cutoffs around 4-10 Hz are commonly used for gait and movement-level muscle activation profiles.",
        "normalization": "Subject-level percentile or MVC normalization reduces inter-subject amplitude scaling effects and should be recorded for every sample.",
    }

    @classmethod
    def get(cls, stage: str) -> str:
        return cls.LITERATURE_MAP.get(stage, "Accepted offline surface-EMG preprocessing protocol.")
