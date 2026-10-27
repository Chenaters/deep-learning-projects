"""Shared project configuration."""

from pathlib import Path

FEATURE_COLUMNS = (
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "motor_current_a",
    "throughput_units_h",
)

LABEL_COLUMN = "is_anomaly"
ANOMALY_TYPE_COLUMN = "anomaly_type"
DEFAULT_RANDOM_SEED = 42
DEFAULT_THRESHOLD_QUANTILE = 0.985

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

