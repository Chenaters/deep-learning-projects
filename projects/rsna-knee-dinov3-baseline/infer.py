from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_dinov3.config import DataConfig, ModelConfig
from rsna_dinov3.constants import LABELS
from rsna_dinov3.data import KneeStudyDataset
from rsna_dinov3.engine import predict
from rsna_dinov3.model import DinoStudyClassifier
from rsna_dinov3.utils import default_workers, resolve_device, seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv3 fold-ensemble inference")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths or glob patterns, e.g. outputs/dinov3_fold*.pt",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override checkpoint model path (required for an offline Kaggle model input)",
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def expand_checkpoints(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(item) for item in glob.glob(pattern))
        paths.extend(matches or [Path(pattern)])
    unique = list(dict.fromkeys(paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint(s) not found: {missing}")
    return unique


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint_paths = expand_checkpoints(args.checkpoints)
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in checkpoint_paths]

    if any(checkpoint["labels"] != LABELS for checkpoint in checkpoints):
        raise ValueError("Checkpoint label order does not match the competition")
    first_data_config = checkpoints[0]["data_config"]
    if any(checkpoint["data_config"] != first_data_config for checkpoint in checkpoints[1:]):
        raise ValueError("All ensemble checkpoints must use the same data configuration")
    data_config = DataConfig(**first_data_config)

    test = pd.read_csv(args.data_dir / "test.csv")
    series = pd.read_csv(args.data_dir / "test_series.csv")
    dataset = KneeStudyDataset(
        test, series, args.data_dir / "test_series", data_config, training=False
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    accumulated_logits: np.ndarray | None = None
    expected_uids: list[str] | None = None
    for path, checkpoint in zip(checkpoint_paths, checkpoints):
        model_values = dict(checkpoint["model_config"])
        if args.model_path:
            model_values["model_path"] = args.model_path
        model = DinoStudyClassifier(ModelConfig(**model_values))
        model.load_trainable_state_dict(checkpoint["head_state_dict"])
        model.to(device)
        uids, logits = predict(model, loader, device)
        if expected_uids is not None and uids != expected_uids:
            raise RuntimeError("Study order changed between ensemble folds")
        expected_uids = uids
        accumulated_logits = logits if accumulated_logits is None else accumulated_logits + logits
        print(f"Loaded {path} (CV AUC={checkpoint['macro_auc']:.5f})")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_logits = np.clip(accumulated_logits / len(checkpoints), -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-mean_logits))
    submission = pd.DataFrame(probabilities, columns=LABELS)
    submission.insert(0, "StudyInstanceUID", expected_uids)
    submission = test[["StudyInstanceUID"]].merge(
        submission, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    if submission[LABELS].isna().any().any():
        raise RuntimeError("Submission contains missing predictions")
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
