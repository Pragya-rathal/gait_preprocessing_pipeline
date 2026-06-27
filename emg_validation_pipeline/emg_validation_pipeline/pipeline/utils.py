import logging
import yaml
from pathlib import Path

def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("EMGValidationPipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        f_handler = logging.FileHandler(output_dir / "preprocessing_audit.log")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    return logger

class BiomechanicalJustification:
    LITERATURE_MAP = {
        "bandpass": "De Luca et al. (2010): 20 Hz high-pass removes aggressive motion artifacts and skin-electrode interface noise without sacrificing true motor unit firing spectra.",
        "notch": "SENIAM Guidelines: 50 Hz notch suppresses ambient AC power line radiation; dual-pass zero-phase filtering preserves true phase timing accuracy.",
        "rectification": "Basmajian & De Luca (1985): Full-wave rectification converts raw high-frequency fluctuations into absolute magnitude values, providing a mathematically valid input for integration.",
        "envelope": "Winter (2009): A low-pass filter at 6 Hz captures the low-frequency mechanical changes of muscle tension, reflecting the gross neural drive profile.",
        "normalization": "Burden et al. (2003): Subject/trial-maximum normalization reduces inter-subject anatomical variance (e.g., subcutaneous fat tissue insulation changes) to isolate functional muscle synergy weights."
    }
    
    @classmethod
    def get(cls, stage: str) -> str:
        return cls.LITERATURE_MAP.get(stage, "Accepted Biomechanical Processing Protocol.")