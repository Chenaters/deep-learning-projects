from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from pydicom.pixels import apply_modality_lut
from torch.utils.data import Dataset

from .config import DataConfig
from .constants import IMAGENET_MEAN, IMAGENET_STD, LABELS, PLANE_TO_ID


def make_folds(train: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    """Stratify both label values and label availability; NaNs remain valid masks."""
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    frame = train.copy()
    known = frame[LABELS].notna().to_numpy(dtype=np.int8)
    positive = frame[LABELS].fillna(0).to_numpy(dtype=np.int8)
    strata = np.concatenate([known, positive], axis=1)
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    frame["fold"] = -1
    dummy = np.zeros(len(frame), dtype=np.int8)
    for fold, (_, valid_idx) in enumerate(splitter.split(dummy, strata)):
        frame.loc[frame.index[valid_idx], "fold"] = fold
    return frame


def calculate_pos_weight(frame: pd.DataFrame, max_weight: float = 10.0) -> torch.Tensor:
    values = frame[LABELS].to_numpy(dtype=np.float32)
    positives = np.nansum(values == 1, axis=0).astype(np.float32)
    negatives = np.nansum(values == 0, axis=0).astype(np.float32)
    weights = negatives / np.maximum(positives, 1.0)
    return torch.from_numpy(np.clip(weights, 1.0, max_weight))


def _dicom_position(path: Path) -> tuple[float, float]:
    """Return a geometry-aware sort coordinate and InstanceNumber fallback."""
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber"],
        )
        fallback = float(getattr(ds, "InstanceNumber", 0))
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
        position = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
        normal = np.cross(orientation[:3], orientation[3:])
        return float(np.dot(position, normal)), fallback
    except Exception:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["InstanceNumber"])
            return float(getattr(ds, "InstanceNumber", 0)), 0.0
        except Exception:
            return 0.0, 0.0


def sorted_dicoms(series_dir: Path) -> list[Path]:
    paths = list(series_dir.glob("*.dcm"))
    return sorted(paths, key=lambda path: (*_dicom_position(path), path.name))


def _sample_centers(length: int, count: int, training: bool) -> np.ndarray:
    if length <= 0:
        return np.zeros(count, dtype=np.int64)
    edges = np.linspace(0, length, count + 1)
    indices: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        lo = min(int(math.floor(left)), length - 1)
        hi = min(max(int(math.ceil(right)) - 1, lo), length - 1)
        if training and hi > lo:
            indices.append(int(np.random.randint(lo, hi + 1)))
        else:
            indices.append((lo + hi) // 2)
    return np.asarray(indices, dtype=np.int64)


def _read_pixels(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    image = np.asarray(apply_modality_lut(ds.pixel_array, ds), dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D DICOM slice, got {image.shape} in {path}")
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image
    return image


def _resize_and_pad(image: torch.Tensor, size: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    scale = size / max(height, width)
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))
    image = F.interpolate(
        image[None, None], size=(new_h, new_w), mode="bilinear", align_corners=False
    )[0, 0]
    pad_h, pad_w = size - new_h, size - new_w
    return F.pad(
        image,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
    )


def decode_series(
    series_dir: Path,
    config: DataConfig,
    training: bool,
) -> torch.Tensor:
    """Decode K 2.5D images; adjacent MRI slices form the RGB-like channels."""
    paths = sorted_dicoms(series_dir)
    if not paths:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")

    centers = _sample_centers(len(paths), config.slices_per_series, training)
    triplets = [
        (max(0, idx - 1), idx, min(len(paths) - 1, idx + 1)) for idx in centers
    ]
    required = sorted({idx for triplet in triplets for idx in triplet})
    decoded = {idx: _read_pixels(paths[idx]) for idx in required}

    all_values = np.concatenate([decoded[idx].ravel() for idx in required])
    low, high = np.percentile(
        all_values, [config.percentile_low, config.percentile_high]
    )
    scale = max(float(high - low), 1e-6)
    normalized = {
        idx: _resize_and_pad(
            torch.from_numpy(np.clip((image - low) / scale, 0.0, 1.0)),
            config.image_size,
        )
        for idx, image in decoded.items()
    }
    images = torch.stack(
        [torch.stack([normalized[idx] for idx in triplet]) for triplet in triplets]
    )
    return images.float()


def _round_robin_series(rows: pd.DataFrame, limit: int, training: bool) -> pd.DataFrame:
    if len(rows) <= limit:
        return rows.sample(frac=1.0) if training and len(rows) > 1 else rows

    groups: list[list[int]] = []
    for plane in ("Sagittal", "Coronal", "Axial"):
        indices = rows.index[rows["Anatomical_Plane"] == plane].tolist()
        if training:
            np.random.shuffle(indices)
        else:
            indices.sort(key=lambda idx: str(rows.at[idx, "SeriesInstanceUID"]))
        groups.append(indices)
    other = rows.index[~rows["Anatomical_Plane"].isin(PLANE_TO_ID)].tolist()
    groups.append(other)

    chosen: list[int] = []
    while len(chosen) < limit and any(groups):
        for group in groups:
            if group and len(chosen) < limit:
                chosen.append(group.pop(0))
    return rows.loc[chosen]


class KneeStudyDataset(Dataset):
    def __init__(
        self,
        studies: pd.DataFrame,
        series: pd.DataFrame,
        dicom_root: str | Path,
        config: DataConfig,
        training: bool,
    ) -> None:
        self.studies = studies.reset_index(drop=True)
        self.series_by_study = {
            uid: group.reset_index(drop=True)
            for uid, group in series.groupby("StudyInstanceUID", sort=False)
        }
        self.dicom_root = Path(dicom_root)
        self.config = config
        self.training = training
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.studies)

    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            images = images.flip(-1)
        if torch.rand(()) < 0.35:
            gamma = float(torch.empty(1).uniform_(0.85, 1.15))
            images = images.clamp(0, 1).pow(gamma)
        return images

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.studies.iloc[index]
        uid = str(row["StudyInstanceUID"])
        rows = self.series_by_study.get(uid)
        if rows is None or rows.empty:
            raise FileNotFoundError(f"No series metadata for study {uid}")
        rows = _round_robin_series(rows, self.config.max_series, self.training)

        max_images = self.config.max_images
        images = torch.zeros(max_images, 3, self.config.image_size, self.config.image_size)
        plane_ids = torch.full((max_images,), 3, dtype=torch.long)
        series_meta = torch.zeros(max_images, 2)
        valid = torch.zeros(max_images, dtype=torch.bool)

        offset = 0
        errors: list[str] = []
        for _, series_row in rows.iterrows():
            if offset >= max_images:
                break
            series_uid = str(series_row["SeriesInstanceUID"])
            series_dir = self.dicom_root / uid / series_uid
            try:
                decoded = decode_series(series_dir, self.config, self.training)
            except Exception as exc:
                errors.append(f"{series_uid}: {exc}")
                continue
            count = min(len(decoded), max_images - offset)
            slot = slice(offset, offset + count)
            images[slot] = decoded[:count]
            plane_ids[slot] = PLANE_TO_ID.get(str(series_row["Anatomical_Plane"]), 3)
            fluid = series_row.get("Fluid_Sensitive", 0)
            fat_suppression = series_row.get("Fat_Suppression", 0)
            series_meta[slot, 0] = 0.0 if pd.isna(fluid) else float(fluid)
            series_meta[slot, 1] = (
                0.0 if pd.isna(fat_suppression) else float(fat_suppression)
            )
            valid[slot] = True
            offset += count

        if not valid.any():
            details = "; ".join(errors[:3])
            raise RuntimeError(f"Could not decode any series for study {uid}. {details}")
        if self.training:
            images[valid] = self._augment(images[valid])
        images[valid] = (images[valid] - self.mean) / self.std

        target = torch.zeros(len(LABELS), dtype=torch.float32)
        target_mask = torch.zeros(len(LABELS), dtype=torch.bool)
        if all(label in row.index for label in LABELS):
            values = row[LABELS].to_numpy(dtype=np.float32)
            target_mask = torch.from_numpy(np.isfinite(values))
            target = torch.from_numpy(np.nan_to_num(values, nan=0.0))

        return {
            "images": images,
            "plane_ids": plane_ids,
            "series_meta": series_meta,
            "valid": valid,
            "target": target,
            "target_mask": target_mask,
            "uid": uid,
        }
