from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_data_summary(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    feature_df = df.drop(columns=[target_column])
    correlation_to_target = df.corr(numeric_only=True)[target_column].drop(target_column)
    strongest_feature = correlation_to_target.abs().idxmax()
    summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "missing_values": {key: int(value) for key, value in df.isna().sum().to_dict().items()},
        "class_distribution": {str(key): int(value) for key, value in df[target_column].value_counts().sort_index().to_dict().items()},
        "describe": json.loads(feature_df.describe().round(4).to_json()),
        "correlation_to_target": {key: round(float(value), 4) for key, value in correlation_to_target.to_dict().items()},
        "strongest_feature": strongest_feature,
        "strongest_feature_comment": (
            f"{strongest_feature} ozelligi hedef ile en guclu dogrusal iliskiyi gosteriyor "
            f"({correlation_to_target[strongest_feature]:.4f})."
        ),
    }
    return summary


def split_dataset(df: pd.DataFrame, target_column: str, seed: int, test_size: float, validation_size_within_train: float) -> DataSplits:
    X = df.drop(columns=[target_column]).to_numpy(dtype=np.float64)
    y = df[target_column].to_numpy(dtype=np.int64).reshape(-1, 1)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_size_within_train,
        random_state=seed,
        stratify=y_train_val,
    )
    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=df.drop(columns=[target_column]).columns.tolist(),
    )


def prepare_features(splits: DataSplits, preprocessing: str) -> tuple[dict[str, np.ndarray], StandardScaler | None]:
    if preprocessing not in {"raw", "standardized"}:
        raise ValueError(f"Unsupported preprocessing: {preprocessing}")
    if preprocessing == "raw":
        prepared = {
            "X_train": splits.X_train.copy(),
            "X_val": splits.X_val.copy(),
            "X_test": splits.X_test.copy(),
        }
        return prepared, None
    scaler = StandardScaler()
    X_train = scaler.fit_transform(splits.X_train)
    X_val = scaler.transform(splits.X_val)
    X_test = scaler.transform(splits.X_test)
    prepared = {
        "X_train": X_train.astype(np.float64),
        "X_val": X_val.astype(np.float64),
        "X_test": X_test.astype(np.float64),
    }
    return prepared, scaler
