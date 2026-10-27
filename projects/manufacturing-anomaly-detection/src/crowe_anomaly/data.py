"""Synthetic manufacturing telemetry for safe, reproducible practice."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANOMALY_TYPE_COLUMN, DEFAULT_RANDOM_SEED, LABEL_COLUMN


def generate_manufacturing_data(
    n_rows: int = 6000,
    anomaly_rate: float = 0.08,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate correlated sensor data and inject labeled failure patterns."""
    if n_rows < 200:
        raise ValueError("n_rows must be at least 200 for stable train/validation/test splits.")
    if not 0 < anomaly_rate < 0.4:
        raise ValueError("anomaly_rate must be between 0 and 0.4.")

    rng = np.random.default_rng(seed)
    load = rng.normal(0.0, 1.0, n_rows)
    ambient = rng.normal(0.0, 1.0, n_rows)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=n_rows, freq="5min", tz="UTC"
            ).astype(str),
            "batch_id": [f"BATCH-{index // 120:04d}" for index in range(n_rows)],
            "temperature_c": 71.0 + 7.5 * load + 1.8 * ambient + rng.normal(0, 1.2, n_rows),
            "vibration_mm_s": 2.1 + 0.45 * np.abs(load) + rng.normal(0, 0.18, n_rows),
            "pressure_bar": 35.0 + 2.8 * load + rng.normal(0, 0.7, n_rows),
            "motor_current_a": 12.5 + 2.4 * load + rng.normal(0, 0.5, n_rows),
            "throughput_units_h": 102.0 + 13.0 * load - 1.5 * ambient + rng.normal(0, 2.5, n_rows),
        }
    )
    frame[LABEL_COLUMN] = 0
    frame[ANOMALY_TYPE_COLUMN] = "normal"

    anomaly_count = max(1, int(round(n_rows * anomaly_rate)))
    eligible = np.arange(max(20, n_rows // 20), n_rows)
    anomaly_indices = rng.choice(eligible, size=anomaly_count, replace=False)
    anomaly_types = rng.choice(
        ["bearing_failure", "overheating", "pressure_leak"],
        size=anomaly_count,
        p=[0.38, 0.34, 0.28],
    )

    for index, anomaly_type in zip(anomaly_indices, anomaly_types, strict=True):
        frame.at[index, LABEL_COLUMN] = 1
        frame.at[index, ANOMALY_TYPE_COLUMN] = anomaly_type
        if anomaly_type == "bearing_failure":
            frame.at[index, "vibration_mm_s"] += rng.uniform(2.4, 5.5)
            frame.at[index, "motor_current_a"] += rng.uniform(2.0, 4.0)
            frame.at[index, "throughput_units_h"] -= rng.uniform(8.0, 18.0)
        elif anomaly_type == "overheating":
            frame.at[index, "temperature_c"] += rng.uniform(18.0, 34.0)
            frame.at[index, "motor_current_a"] += rng.uniform(1.0, 3.0)
        else:
            frame.at[index, "pressure_bar"] -= rng.uniform(9.0, 16.0)
            frame.at[index, "throughput_units_h"] -= rng.uniform(12.0, 25.0)

    return frame


def split_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological 60/20/20 splits to avoid future-data leakage."""
    first_cut = int(len(frame) * 0.60)
    second_cut = int(len(frame) * 0.80)
    return (
        frame.iloc[:first_cut].copy(),
        frame.iloc[first_cut:second_cut].copy(),
        frame.iloc[second_cut:].copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--anomaly-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=Path("data/manufacturing.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_manufacturing_data(args.rows, args.anomaly_rate, args.seed)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows to {args.output}")


if __name__ == "__main__":
    main()

