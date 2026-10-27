#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

TARGET_ORDER = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_small_dino_model() -> str:
    available = set(timm.list_models("*dino*"))
    preferred = [
        "vit_small_patch16_dinov3",
        "vit_small_patch16_dinov3_qkvb",
        "vit_small_plus_patch16_dinov3",
        "vit_small_plus_patch16_dinov3_qkvb",
        "vit_small_patch14_dinov2.lvd142m",
        "vit_small_patch14_reg4_dinov2.lvd142m",
        "vit_small_patch16_224.dino",
        "vit_small_patch8_224.dino",
        "vit_small_patch14_dinov2",
        "vit_small_patch14_reg4_dinov2",
    ]

    for name in preferred:
        if name in available:
            return name

    fallback = sorted(
        [
            name
            for name in available
            if "small" in name and ("dinov2" in name or ".dino" in name)
        ]
    )
    if fallback:
        return fallback[0]

    raise ValueError("No small DINO model found in this timm install.")


class BiomassImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_dir: Path,
        image_col: str,
        target_cols: list[str],
        transform,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.image_col = image_col
        self.target_cols = target_cols
        self.transform = transform
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = self.dataset_dir / row[self.image_col]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = row[self.target_cols].values.astype(np.float32)
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std

        return image, torch.from_numpy(target)


class DinoRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_targets: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features

        if hidden_dim <= 0:
            self.regressor = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_targets),
            )
        else:
            self.regressor = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_targets),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.regressor(features)


def infer_head_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, float]:
    if "regressor.1.weight" in state_dict and "regressor.4.weight" in state_dict:
        hidden_dim = int(state_dict["regressor.1.weight"].shape[0])
        return hidden_dim, 0.3

    return 0, 0.2


def prepare_train_dataframe(train_csv: Path) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv)
    pivoted = train_df.pivot_table(
        index=[
            "image_path",
            "Sampling_Date",
            "State",
            "Species",
            "Pre_GSHH_NDVI",
            "Height_Ave_cm",
        ],
        columns="target_name",
        values="target",
    ).reset_index()

    missing = [target for target in TARGET_ORDER if target not in pivoted.columns]
    if missing:
        raise ValueError(f"Missing targets in pivoted data: {missing}")

    return pivoted


def build_eval_transform(model_name: str):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(model_name, pretrained=False)
    )
    return timm.data.create_transform(
        input_size=data_cfg["input_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        interpolation=data_cfg["interpolation"],
        is_training=False,
    )


def make_val_loader(
    dataset_dir: Path,
    model_name: str,
    seed: int,
    val_ratio: float,
    batch_size: int,
    num_workers: int,
    target_mean: np.ndarray | None,
    target_std: np.ndarray | None,
) -> DataLoader:
    train_pivoted = prepare_train_dataframe(dataset_dir / "train.csv")

    state_counts = train_pivoted["State"].value_counts()
    can_stratify = len(state_counts) > 1 and state_counts.min() >= 2
    stratify_col = train_pivoted["State"] if can_stratify else None

    _, val_data = train_test_split(
        train_pivoted,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify_col,
    )

    eval_transform = build_eval_transform(model_name)
    val_dataset = BiomassImageDataset(
        dataframe=val_data,
        dataset_dir=dataset_dir,
        image_col="image_path",
        target_cols=TARGET_ORDER,
        transform=eval_transform,
        target_mean=target_mean,
        target_std=target_std,
    )

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[float, float, float]:
    criterion = nn.SmoothL1Loss(beta=2.0)
    model.eval()

    running_loss = 0.0
    mae_sum = 0.0
    mse_sum = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validate", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            outputs_denorm = outputs * target_std + target_mean
            targets_denorm = targets * target_std + target_mean
            mae = torch.mean(torch.abs(outputs_denorm - targets_denorm))
            mse = torch.mean((outputs_denorm - targets_denorm) ** 2)

            batch_n = images.size(0)
            running_loss += loss.item() * batch_n
            mae_sum += mae.item() * batch_n
            mse_sum += mse.item() * batch_n

    val_loss = running_loss / len(dataloader.dataset)
    val_mae = mae_sum / len(dataloader.dataset)
    val_rmse = (mse_sum / len(dataloader.dataset)) ** 0.5
    return val_loss, val_mae, val_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DINO checkpoint on validation split")
    parser.add_argument("--checkpoint", type=Path, default=Path("dino_small_model_best.pt"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("csiro-biomass"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model_name = checkpoint.get("model_name") or pick_small_dino_model()

    hidden_dim, dropout = infer_head_from_state_dict(state_dict)
    model = DinoRegressor(
        model_name=model_name,
        num_targets=len(TARGET_ORDER),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)

    has_norm = "target_mean" in checkpoint and "target_std" in checkpoint
    if has_norm:
        target_mean_np = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        target_std_np = np.asarray(checkpoint["target_std"], dtype=np.float32)
    else:
        target_mean_np = None
        target_std_np = None

    val_loader = make_val_loader(
        dataset_dir=args.dataset_dir,
        model_name=model_name,
        seed=args.seed,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_mean=target_mean_np,
        target_std=target_std_np,
    )

    if target_mean_np is None or target_std_np is None:
        target_mean_t = torch.zeros(len(TARGET_ORDER), dtype=torch.float32, device=device)
        target_std_t = torch.ones(len(TARGET_ORDER), dtype=torch.float32, device=device)
    else:
        target_mean_t = torch.tensor(target_mean_np, dtype=torch.float32, device=device)
        target_std_t = torch.tensor(target_std_np, dtype=torch.float32, device=device)

    val_loss, val_mae, val_rmse = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        target_mean=target_mean_t,
        target_std=target_std_t,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {model_name}")
    print(f"Normalization from checkpoint: {has_norm}")
    print(f"Validation loss (SmoothL1): {val_loss:.6f}")
    print(f"Validation MAE: {val_mae:.6f}")
    print(f"Validation RMSE: {val_rmse:.6f}")


if __name__ == "__main__":
    main()
