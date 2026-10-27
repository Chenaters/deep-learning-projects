"""Model fitting, candidate comparison, scoring, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_THRESHOLD_QUANTILE, FEATURE_COLUMNS, LABEL_COLUMN
from .schema import validate_frame


@dataclass(frozen=True)
class CandidateResult:
    name: str
    model: MLPRegressor
    threshold: float
    validation_metrics: dict[str, Any]


def reconstruction_error(
    scaler: StandardScaler, model: MLPRegressor, feature_frame: pd.DataFrame
) -> np.ndarray:
    clean = validate_frame(feature_frame)
    scaled = scaler.transform(clean)
    reconstructed = model.predict(scaled)
    return np.mean(np.square(scaled - reconstructed), axis=1)


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold": float(threshold),
        "flag_rate": round(float(predictions.mean()), 4),
    }


def fit_autoencoder_candidates(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    seed: int,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    max_iter: int = 220,
    architectures: tuple[tuple[int, ...], ...] = ((16, 5, 16), (24, 8, 24)),
) -> tuple[StandardScaler, CandidateResult, list[dict[str, Any]]]:
    train_normal = train_frame.loc[train_frame[LABEL_COLUMN] == 0]
    scaler = StandardScaler().fit(validate_frame(train_normal))
    train_scaled = scaler.transform(validate_frame(train_normal))

    validation_normal = validation_frame.loc[validation_frame[LABEL_COLUMN] == 0]
    validation_labels = validation_frame[LABEL_COLUMN].to_numpy(dtype=int)
    comparisons: list[dict[str, Any]] = []
    results: list[CandidateResult] = []

    for architecture in architectures:
        model = MLPRegressor(
            hidden_layer_sizes=architecture,
            activation="relu",
            solver="adam",
            alpha=0.0005,
            batch_size=64,
            learning_rate_init=0.002,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
        model.fit(train_scaled, train_scaled)
        normal_scores = reconstruction_error(scaler, model, validation_normal)
        threshold = float(np.quantile(normal_scores, threshold_quantile))
        all_scores = reconstruction_error(scaler, model, validation_frame)
        metrics = classification_metrics(validation_labels, all_scores, threshold)
        name = "autoencoder_" + "x".join(str(size) for size in architecture)
        comparisons.append({"candidate": name, "iterations": int(model.n_iter_), **metrics})
        results.append(CandidateResult(name, model, threshold, metrics))

    best = max(
        results,
        key=lambda result: (
            result.validation_metrics["f1"],
            result.validation_metrics["recall"],
            -result.validation_metrics["false_positives"],
        ),
    )
    return scaler, best, comparisons


def isolation_forest_baseline(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    seed: int,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
) -> dict[str, Any]:
    train_normal = train_frame.loc[train_frame[LABEL_COLUMN] == 0]
    scaler = StandardScaler().fit(validate_frame(train_normal))
    train_scaled = scaler.transform(validate_frame(train_normal))
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(train_scaled)

    validation_normal = validation_frame.loc[validation_frame[LABEL_COLUMN] == 0]
    normal_scores = -model.score_samples(scaler.transform(validate_frame(validation_normal)))
    threshold = float(np.quantile(normal_scores, threshold_quantile))
    test_scores = -model.score_samples(scaler.transform(validate_frame(test_frame)))
    return {
        "model": "isolation_forest_baseline",
        **classification_metrics(
            test_frame[LABEL_COLUMN].to_numpy(dtype=int), test_scores, threshold
        ),
    }


def score_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    scores = reconstruction_error(bundle["scaler"], bundle["model"], frame)
    result = frame.copy()
    result["anomaly_score"] = scores
    result["predicted_anomaly"] = (scores >= bundle["threshold"]).astype(int)
    return result


def metrics_by_anomaly_type(scored_frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for name, group in scored_frame.groupby("anomaly_type"):
        output[str(name)] = {
            "rows": int(len(group)),
            "flagged": int(group["predicted_anomaly"].sum()),
        }
    return output

