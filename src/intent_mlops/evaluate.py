"""Model evaluation helpers shared by training, tests and monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_proba(model, features: pd.DataFrame) -> np.ndarray:
    """Positive-class probability for a batch of features."""
    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


def binary_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | list[list[int]]]:
    """Core classification metrics at a decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "confusion_matrix": matrix,
    }


def best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return the pair that maximizes F1."""
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.1, 0.91, 0.05):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def evaluate_model(model, features: pd.DataFrame, target: pd.Series) -> dict:
    """Full evaluation report at 0.5 and at the best-F1 threshold."""
    probabilities = predict_proba(model, features)
    report = binary_metrics(target, probabilities)
    threshold, f1 = best_f1_threshold(target, probabilities)
    report["best_f1_threshold"] = threshold
    report["best_f1"] = f1
    report["best_threshold_metrics"] = binary_metrics(target, probabilities, threshold)
    report["positive_rate"] = float(target.mean())
    return report
