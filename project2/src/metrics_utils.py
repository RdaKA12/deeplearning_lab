from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_evaluation_bundle(y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )
    per_class_f1 = {class_name: float(report[class_name]["f1-score"]) for class_name in class_names}
    return {
        **compute_multiclass_metrics(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
        "per_class_f1": per_class_f1,
    }
