"""Train, compare, evaluate, and package the anomaly detector."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_RANDOM_SEED,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
)
from .data import generate_manufacturing_data, split_by_time
from .modeling import (
    classification_metrics,
    fit_autoencoder_candidates,
    isolation_forest_baseline,
    metrics_by_anomaly_type,
    reconstruction_error,
    score_frame,
)
from .report import write_metrics, write_stakeholder_summary


def train_pipeline(
    *,
    data_dir: Path,
    output_dir: Path,
    rows: int = 6000,
    seed: int = DEFAULT_RANDOM_SEED,
    max_iter: int = 220,
    architectures: tuple[tuple[int, ...], ...] = ((16, 5, 16), (24, 8, 24)),
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_frame = generate_manufacturing_data(n_rows=rows, seed=seed)
    train_frame, validation_frame, test_frame = split_by_time(full_frame)
    full_frame.to_csv(data_dir / "manufacturing_telemetry.csv", index=False)

    scaler, best, candidate_metrics = fit_autoencoder_candidates(
        train_frame,
        validation_frame,
        seed=seed,
        max_iter=max_iter,
        architectures=architectures,
    )
    test_scores = reconstruction_error(scaler, best.model, test_frame)
    test_metrics = classification_metrics(
        test_frame[LABEL_COLUMN].to_numpy(dtype=int), test_scores, best.threshold
    )

    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    bundle = {
        "model_version": "0.1.0",
        "created_at": created_at,
        "model_name": best.name,
        "features": list(FEATURE_COLUMNS),
        "threshold": best.threshold,
        "scaler": scaler,
        "model": best.model,
    }
    joblib.dump(bundle, output_dir / "model.joblib")

    scored_test = score_frame(bundle, test_frame)
    scored_test.to_csv(output_dir / "test_scored.csv", index=False)
    metrics: dict[str, Any] = {
        "created_at": created_at,
        "data": {
            "rows": int(len(full_frame)),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "test_rows": int(len(test_frame)),
            "test_anomalies": int(test_frame[LABEL_COLUMN].sum()),
        },
        "selected_candidate": best.name,
        "candidate_validation": candidate_metrics,
        "autoencoder_test": test_metrics,
        "baseline_test": isolation_forest_baseline(
            train_frame, validation_frame, test_frame, seed=seed
        ),
        "test_flags_by_type": metrics_by_anomaly_type(scored_test),
    }
    write_metrics(metrics, output_dir / "metrics.json")
    write_stakeholder_summary(metrics, output_dir / "stakeholder_summary.md")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--max-iter", type=int, default=220)
    args = parser.parse_args()

    metrics = train_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rows=args.rows,
        seed=args.seed,
        max_iter=args.max_iter,
    )
    test = metrics["autoencoder_test"]
    print(
        f"Selected {metrics['selected_candidate']} | "
        f"precision={test['precision']:.3f} recall={test['recall']:.3f} "
        f"f1={test['f1']:.3f}"
    )


if __name__ == "__main__":
    main()

