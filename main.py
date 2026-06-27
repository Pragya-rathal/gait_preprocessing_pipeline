#!/usr/bin/env python3
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parent / "emg_validation_pipeline" / "emg_validation_pipeline"
sys.path.insert(0, str(PACKAGE_ROOT))

from main import execute_verification_pipeline  # type: ignore  # noqa: E402

if __name__ == "__main__":
    execute_verification_pipeline()
