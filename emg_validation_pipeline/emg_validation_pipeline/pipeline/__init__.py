"""EMG preprocessing and publication dataset generation package."""
from .conditioning import ScientificEMGConditioner
from .loader import C3DValidationLoader
from .normalization import ConfigurableNormalizer
from .quality_control import SignalQualityAuditor
from .windowing import ResearchWindowSlicer, CausalWindowSlicer

__all__ = [
    "ScientificEMGConditioner",
    "C3DValidationLoader",
    "ConfigurableNormalizer",
    "SignalQualityAuditor",
    "ResearchWindowSlicer",
    "CausalWindowSlicer",
]
