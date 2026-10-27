#!/usr/bin/env python
# coding: utf-8

# # CSIRO Biomass: Small DINO Notebook
# 
# This notebook sets up a compact DINO-based regressor for the CSIRO biomass competition using a small backbone when available.

# In[ ]:


# If needed, install dependencies in this environment:
# !pip install torch torchvision timm pandas numpy scikit-learn pillow tqdm


# In[1]:


from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# In[2]:


TARGET_ORDER = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


@dataclass
class Config:
    dataset_dir: Path = Path("csiro-biomass")
    model_name: str | None = None
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 30
    freeze_epochs: int = 2
    warmup_epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    use_pretrained_backbone: bool = True
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    comparison_csv: Path = Path("results/model_comparison_from_saved_models.csv")
    output_model: Path = Path("dino_small_model_best.pt")
    output_submission: Path = Path("dino_small_submission.csv")


cfg = Config()
cfg


# In[3]:


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

    raise ValueError(
        "No small DINO model found in this timm install. "
        "Run timm.list_models('*dino*') to inspect what is available."
    )


if cfg.model_name is None:
    cfg.model_name = pick_small_dino_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(cfg.seed)

print(f"Using device: {device}")
print(f"Using model: {cfg.model_name}")
print(f"Using pretrained backbone weights: {cfg.use_pretrained_backbone}")


# In[4]:


class BiomassImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dataset_dir: Path,
        image_col: str,
        target_cols: list[str] | None,
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

        if self.target_cols is None:
            return image

        target = row[self.target_cols].values.astype(np.float32)
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std
        return image, torch.from_numpy(target)


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


# In[5]:


def build_transforms(model_name: str, image_size: int):
    data_cfg = timm.data.resolve_model_data_config(
        timm.create_model(model_name, pretrained=False)
    )
    model_input_size = data_cfg.get("input_size", (3, image_size, image_size))
    if not isinstance(model_input_size, (tuple, list)) or len(model_input_size) != 3:
        model_input_size = (3, image_size, image_size)

    eval_transform = timm.data.create_transform(
        input_size=model_input_size,
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        interpolation=data_cfg["interpolation"],
        is_training=False,
    )

    train_transform = timm.data.create_transform(
        input_size=model_input_size,
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        interpolation=data_cfg["interpolation"],
        is_training=True,
        hflip=0.5,
        vflip=0.2,
        color_jitter=0.2,
    )
    return train_transform, eval_transform


def create_dataloaders(cfg: Config, train_pivoted: pd.DataFrame):
    state_counts = train_pivoted["State"].value_counts()
    can_stratify = len(state_counts) > 1 and state_counts.min() >= 2
    stratify_col = train_pivoted["State"] if can_stratify else None

    train_data, val_data = train_test_split(
        train_pivoted,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=stratify_col,
    )

    target_mean = train_data[TARGET_ORDER].mean().values.astype(np.float32)
    target_std = train_data[TARGET_ORDER].std().values.astype(np.float32)
    target_std = np.clip(target_std, 1e-6, None)

    train_transform, eval_transform = build_transforms(cfg.model_name, cfg.image_size)

    train_dataset = BiomassImageDataset(
        dataframe=train_data,
        dataset_dir=cfg.dataset_dir,
        image_col="image_path",
        target_cols=TARGET_ORDER,
        transform=train_transform,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_dataset = BiomassImageDataset(
        dataframe=val_data,
        dataset_dir=cfg.dataset_dir,
        image_col="image_path",
        target_cols=TARGET_ORDER,
        transform=eval_transform,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, eval_transform, train_data, val_data, target_mean, target_std


# In[6]:


class DinoRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_targets: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=use_pretrained_backbone,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features
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

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


# In[7]:


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, targets in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    mae_sum = 0.0
    rmse_sum = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            outputs_denorm = outputs * target_std + target_mean
            targets_denorm = targets * target_std + target_mean
            mae = torch.mean(torch.abs(outputs_denorm - targets_denorm))
            mse = torch.mean((outputs_denorm - targets_denorm) ** 2)

            running_loss += loss.item() * images.size(0)
            mae_sum += mae.item() * images.size(0)
            rmse_sum += mse.item() * images.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    val_mae = mae_sum / len(dataloader.dataset)
    val_rmse = (rmse_sum / len(dataloader.dataset)) ** 0.5
    return val_loss, val_mae, val_rmse


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(0.05, float(cosine))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def upsert_model_result(
    csv_path: Path,
    model_name: str,
    advanced_techniques: str,
    val_loss: float,
    val_mae: float,
    val_rmse: float,
    model_file: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["model", "advanced_techniques", "val_loss", "val_mae", "val_rmse", "model_file"]

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=columns)

    row = {
        "model": model_name,
        "advanced_techniques": advanced_techniques,
        "val_loss": float(val_loss),
        "val_mae": float(val_mae),
        "val_rmse": float(val_rmse),
        "model_file": model_file,
    }

    if "model" in df.columns and (df["model"] == model_name).any():
        df.loc[df["model"] == model_name, list(row.keys())] = list(row.values())
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df = df[columns]
    df.to_csv(csv_path, index=False)
    print(f"Updated comparison metrics at {csv_path}")


def train_model(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_mean: np.ndarray,
    target_std: np.ndarray,
):
    model = DinoRegressor(
        model_name=cfg.model_name,
        num_targets=len(TARGET_ORDER),
        hidden_dim=cfg.head_hidden_dim,
        dropout=cfg.head_dropout,
        use_pretrained_backbone=cfg.use_pretrained_backbone,
    ).to(device)
    criterion = nn.SmoothL1Loss(beta=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = make_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_epochs=cfg.warmup_epochs,
        total_epochs=cfg.epochs,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_std, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_metrics = {"val_loss": float("inf"), "val_mae": float("inf"), "val_rmse": float("inf")}

    for epoch in range(1, cfg.epochs + 1):
        model.set_backbone_trainable(epoch > cfg.freeze_epochs)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_mae, val_rmse = validate(
            model,
            val_loader,
            criterion,
            device,
            target_mean=target_mean_t,
            target_std=target_std_t,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_mae={val_mae:.4f} "
            f"| val_rmse={val_rmse:.4f} "
            f"| lr={lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": cfg.model_name,
                    "targets": TARGET_ORDER,
                    "image_size": cfg.image_size,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "best_metrics": best_metrics,
                },
                cfg.output_model,
            )
            print(f"Saved best checkpoint to {cfg.output_model}")

    checkpoint = torch.load(cfg.output_model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, best_metrics


# In[8]:


train_pivoted = prepare_train_dataframe(cfg.dataset_dir / "train.csv")
train_loader, val_loader, eval_transform, train_data, val_data, target_mean, target_std = create_dataloaders(cfg, train_pivoted)

print(f"Train rows: {len(train_data)}")
print(f"Val rows:   {len(val_data)}")
print(train_pivoted[TARGET_ORDER].describe().T[["mean", "std", "min", "max"]])


# In[9]:


# Train the model (adjust cfg.epochs / cfg.batch_size above for faster or stronger runs).
model, best_metrics = train_model(cfg, train_loader, val_loader, target_mean, target_std)

upsert_model_result(
    csv_path=cfg.comparison_csv,
    model_name=f"DINO Small ({cfg.model_name})",
    advanced_techniques=(
        "DINO small pretrained + target normalization + warmup/cosine + "
        "deeper MLP head + two-stage fine-tuning"
    ),
    val_loss=best_metrics["val_loss"],
    val_mae=best_metrics["val_mae"],
    val_rmse=best_metrics["val_rmse"],
    model_file=cfg.output_model.name,
)


# In[ ]:


def predict_unique_images(
    model: DinoRegressor,
    image_paths,
    dataset_dir: Path,
    transform,
    batch_size: int,
    num_workers: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> pd.DataFrame:
    unique_df = pd.DataFrame({"image_path": list(dict.fromkeys(image_paths))})
    test_dataset = BiomassImageDataset(
        dataframe=unique_df,
        dataset_dir=dataset_dir,
        image_col="image_path",
        target_cols=None,
        transform=transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    preds = []
    model.eval()
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Predict", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds.append(outputs.cpu().numpy())

    pred_values = np.vstack(preds)
    pred_values = pred_values * target_std + target_mean
    pred_values = np.maximum(pred_values, 0.0)
    pred_df = pd.DataFrame(pred_values, columns=TARGET_ORDER)
    pred_df["image_path"] = unique_df["image_path"].values
    return pred_df


def create_submission(
    cfg: Config,
    model: DinoRegressor,
    eval_transform,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> pd.DataFrame:
    test_df = pd.read_csv(cfg.dataset_dir / "test.csv")
    pred_df = predict_unique_images(
        model=model,
        image_paths=test_df["image_path"].values,
        dataset_dir=cfg.dataset_dir,
        transform=eval_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        target_mean=target_mean,
        target_std=target_std,
    )

    pred_lookup = pred_df.set_index("image_path")
    rows = []
    for _, row in test_df.iterrows():
        target = float(pred_lookup.loc[row["image_path"], row["target_name"]])
        rows.append({"sample_id": row["sample_id"], "target": target})

    submission = pd.DataFrame(rows)
    submission.to_csv(cfg.output_submission, index=False)
    print(f"Saved submission to {cfg.output_submission} | shape={submission.shape}")
    return submission


# In[ ]:


# Generate a Kaggle submission file.
submission = create_submission(cfg, model, eval_transform, target_mean=target_mean, target_std=target_std)
submission.head()

