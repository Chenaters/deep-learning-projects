#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

TARGET_ORDER = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


class BiomassInferenceDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_dir: Path,
        image_col: str,
        transform,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.image_col = image_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_path = self.dataset_dir / row[self.image_col]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


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


def infer_head_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, float]:
    if (
        "regressor.1.weight" in state_dict
        and "regressor.4.weight" in state_dict
    ):
        hidden_dim = int(state_dict["regressor.1.weight"].shape[0])
        dropout = 0.3
        return hidden_dim, dropout

    return 0, 0.2


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


def predict_unique_images(
    model: nn.Module,
    image_paths: pd.Series,
    dataset_dir: Path,
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> pd.DataFrame:
    unique_df = pd.DataFrame(
        {"image_path": list(dict.fromkeys(image_paths.tolist()))}
    )

    dataset = BiomassInferenceDataset(
        dataframe=unique_df,
        dataset_dir=dataset_dir,
        image_col="image_path",
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    predictions = []
    model.eval()
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Predict", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())

    pred_values = np.vstack(predictions)
    pred_values = pred_values * target_std + target_mean
    pred_values = np.maximum(pred_values, 0.0)

    pred_df = pd.DataFrame(pred_values, columns=TARGET_ORDER)
    pred_df["image_path"] = unique_df["image_path"].values
    return pred_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission using a saved DINO checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("dino_small_model_best.pt"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("csiro-biomass"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dino_small_submission.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    test_csv = args.test_csv or (args.dataset_dir / "test.csv")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
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

    if "target_mean" in checkpoint and "target_std" in checkpoint:
        target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    else:
        target_mean = np.zeros(len(TARGET_ORDER), dtype=np.float32)
        target_std = np.ones(len(TARGET_ORDER), dtype=np.float32)

    eval_transform = build_eval_transform(model_name)

    test_df = pd.read_csv(test_csv)
    required_cols = {"sample_id", "image_path", "target_name"}
    missing = required_cols.difference(test_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in test CSV: {sorted(missing)}"
        )

    pred_df = predict_unique_images(
        model=model,
        image_paths=test_df["image_path"],
        dataset_dir=args.dataset_dir,
        transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
    )

    pred_lookup = pred_df.set_index("image_path")
    submission_rows = []
    for _, row in test_df.iterrows():
        image_path = row["image_path"]
        target_name = row["target_name"]
        if target_name not in TARGET_ORDER:
            raise ValueError(
                f"Unknown target_name '{target_name}' found in test CSV."
            )

        target_value = float(pred_lookup.loc[image_path, target_name])
        submission_rows.append(
            {"sample_id": row["sample_id"], "target": target_value}
        )

    submission = pd.DataFrame(submission_rows)
    submission.to_csv(args.output, index=False)

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Rows written: {len(submission)}")
    print(f"Saved submission to: {args.output}")


if __name__ == "__main__":
    main()
