from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_dinov3.config import DataConfig, ModelConfig
from rsna_dinov3.constants import LABELS
from rsna_dinov3.data import KneeStudyDataset, calculate_pos_weight, make_folds
from rsna_dinov3.engine import evaluate, train_one_epoch
from rsna_dinov3.model import DinoStudyClassifier
from rsna_dinov3.utils import default_workers, ensure_directory, resolve_device, seed_everything, seed_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RSNA DINOv3 baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face model id or a local snapshot directory",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-series", type=int, default=6)
    parser.add_argument("--slices-per-series", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_loader(dataset, args: argparse.Namespace, training: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": training,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def cosine_scheduler(optimizer, epochs: int, updates_per_epoch: int):
    total_steps = max(epochs * updates_per_epoch, 1)
    warmup_steps = max(round(0.1 * total_steps), 1)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_directory(args.output_dir)

    train_csv = args.data_dir / "train.csv"
    series_csv = args.data_dir / "train_series.csv"
    dicom_root = args.data_dir / "train_series"
    train = pd.read_csv(train_csv)
    missing = [column for column in ["StudyInstanceUID", *LABELS] if column not in train]
    if missing:
        raise ValueError(f"Missing train.csv columns: {missing}")
    train = train.loc[train[LABELS].notna().any(axis=1)].reset_index(drop=True)
    train = make_folds(train, args.n_splits, args.seed)
    series = pd.read_csv(series_csv)
    train[["StudyInstanceUID", "fold", *LABELS]].to_csv(
        output_dir / "folds.csv", index=False
    )

    data_config = DataConfig(
        image_size=args.image_size,
        max_series=args.max_series,
        slices_per_series=args.slices_per_series,
    )
    model_config = ModelConfig(model_path=args.model_path, dropout=args.dropout)
    print(
        json.dumps(
            {
                "device": str(device),
                "labeled_studies": len(train),
                "data_config": data_config.to_dict(),
                "model_config": model_config.to_dict(),
            },
            indent=2,
        )
    )

    for fold in args.folds:
        if fold < 0 or fold >= args.n_splits:
            raise ValueError(f"Fold {fold} is outside [0, {args.n_splits - 1}]")
        seed_everything(args.seed + fold)
        train_frame = train.loc[train["fold"] != fold].reset_index(drop=True)
        valid_frame = train.loc[train["fold"] == fold].reset_index(drop=True)
        train_dataset = KneeStudyDataset(
            train_frame, series, dicom_root, data_config, training=True
        )
        valid_dataset = KneeStudyDataset(
            valid_frame, series, dicom_root, data_config, training=False
        )
        train_loader = make_loader(train_dataset, args, training=True)
        valid_loader = make_loader(valid_dataset, args, training=False)

        model = DinoStudyClassifier(model_config).to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, weight_decay=args.weight_decay
        )
        updates = math.ceil(len(train_loader) / args.accumulation_steps)
        scheduler = cosine_scheduler(optimizer, args.epochs, updates)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
        pos_weight = calculate_pos_weight(train_frame).to(device)

        best_auc = -float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                pos_weight,
                args.accumulation_steps,
            )
            valid_loss, valid_auc, label_aucs = evaluate(
                model, valid_loader, device, pos_weight
            )
            metrics = {
                "fold": fold,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "macro_auc": valid_auc,
                "label_auc": dict(zip(LABELS, label_aucs)),
            }
            print(json.dumps(metrics, indent=2))

            if valid_auc > best_auc:
                best_auc = valid_auc
                checkpoint = {
                    "head_state_dict": model.trainable_state_dict(),
                    "model_config": model_config.to_dict(),
                    "data_config": data_config.to_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "macro_auc": valid_auc,
                    "labels": LABELS,
                }
                torch.save(checkpoint, output_dir / f"dinov3_fold{fold}.pt")
        del model, optimizer, scheduler, scaler, train_loader, valid_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

