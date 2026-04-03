from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_evaluation_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        **compute_binary_classification_metrics(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1)).tolist(),
        "classification_report": classification_report(y_true.reshape(-1), y_pred.reshape(-1), zero_division=0, output_dict=True),
    }
