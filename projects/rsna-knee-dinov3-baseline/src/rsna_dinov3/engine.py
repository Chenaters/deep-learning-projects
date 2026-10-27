from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def macro_auc(targets: np.ndarray, predictions: np.ndarray) -> tuple[float, list[float]]:
    scores: list[float] = []
    for index in range(targets.shape[1]):
        known = np.isfinite(targets[:, index])
        values = targets[known, index]
        if len(values) == 0 or len(np.unique(values)) < 2:
            scores.append(float("nan"))
        else:
            scores.append(float(roc_auc_score(values, predictions[known, index])))
    return float(np.nanmean(scores)), scores


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pos_weight: torch.Tensor,
    accumulation_steps: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_items = 0
    use_amp = device.type == "cuda"

    progress = tqdm(loader, desc="train", leave=False)
    for step, raw_batch in enumerate(progress):
        batch = _move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(
                batch["images"], batch["plane_ids"], batch["series_meta"], batch["valid"]
            )
            loss = masked_bce_loss(
                logits, batch["target"], batch["target_mask"], pos_weight
            )
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_step = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        batch_size = batch["images"].shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{total_loss / total_items:.4f}")
    return total_loss / max(total_items, 1)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    pos_weight: torch.Tensor,
) -> tuple[float, float, list[float]]:
    model.eval()
    losses: list[float] = []
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for raw_batch in tqdm(loader, desc="valid", leave=False):
        batch = _move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(
                batch["images"], batch["plane_ids"], batch["series_meta"], batch["valid"]
            )
            loss = masked_bce_loss(
                logits, batch["target"], batch["target_mask"], pos_weight
            )
        targets = batch["target"].float().cpu().numpy()
        mask = batch["target_mask"].cpu().numpy()
        targets[~mask] = np.nan
        losses.append(float(loss))
        all_targets.append(targets)
        all_predictions.append(logits.float().sigmoid().cpu().numpy())

    auc, label_aucs = macro_auc(
        np.concatenate(all_targets), np.concatenate(all_predictions)
    )
    return float(np.mean(losses)), auc, label_aucs


@torch.inference_mode()
def predict(model: torch.nn.Module, loader: Iterable, device: torch.device) -> tuple[list[str], np.ndarray]:
    model.eval()
    uids: list[str] = []
    logits: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    for raw_batch in tqdm(loader, desc="infer", leave=False):
        batch = _move_batch(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(
                batch["images"], batch["plane_ids"], batch["series_meta"], batch["valid"]
            )
        uids.extend(raw_batch["uid"])
        logits.append(output.float().cpu().numpy())
    return uids, np.concatenate(logits)

