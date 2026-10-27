"""Batch-score a CSV with the packaged anomaly detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DEFAULT_ARTIFACT_DIR
from .modeling import score_frame
from .schema import validate_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_ARTIFACT_DIR / "model.joblib"
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    frame = pd.read_csv(args.input)
    validate_frame(frame)
    scored = score_frame(bundle, frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored):,} rows to {args.output}")


if __name__ == "__main__":
    main()

