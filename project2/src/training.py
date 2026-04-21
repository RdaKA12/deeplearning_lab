from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .metrics_utils import build_evaluation_bundle
from .models import count_trainable_parameters


def _move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device, non_blocking: bool) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = batch
    return images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)


def _autocast_context(device_type: str, use_amp: bool) -> Any:
    return torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: Adam,
    criterion: nn.Module,
    device: torch.device,
    runtime: dict[str, Any],
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images, labels = _move_batch(batch, device, runtime["non_blocking"])
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(runtime["device_type"], runtime["use_amp"]):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    runtime: dict[str, Any],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    output_shape: list[int] | None = None

    with torch.no_grad():
        for batch in loader:
            images, labels = _move_batch(batch, device, runtime["non_blocking"])
            with _autocast_context(runtime["device_type"], runtime["use_amp"]):
                logits = model(images)
                loss = criterion(logits, labels)
            if output_shape is None:
                output_shape = list(logits.shape)

            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    bundle = build_evaluation_bundle(y_true_np, y_pred_np, class_names)
    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": bundle["accuracy"],
        "precision": bundle["precision"],
        "recall": bundle["recall"],
        "f1": bundle["f1"],
    }
    return {
        "metrics": metrics,
        "bundle": bundle,
        "output_shape": output_shape or [],
        "y_true": y_true_np,
        "y_pred": y_pred_np,
    }


def train_model(
    experiment_name: str,
    model_family: str,
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    test_loader: DataLoader[Any],
    device: torch.device,
    runtime: dict[str, Any],
    class_names: tuple[str, ...],
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if runtime["use_amp"] else None

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, runtime, scaler)
        val_eval = evaluate_model(model, val_loader, criterion, device, runtime, class_names)

        record = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_eval["metrics"]["loss"],
            "val_accuracy": val_eval["metrics"]["accuracy"],
        }
        history.append(record)
        print(
            f"[{experiment_name}] epoch={epoch} "
            f"train_loss={record['train_loss']:.4f} train_acc={record['train_accuracy']:.4f} "
            f"val_loss={record['val_loss']:.4f} val_acc={record['val_accuracy']:.4f}"
        )

        if val_eval["metrics"]["loss"] < best_val_loss - min_delta:
            best_val_loss = val_eval["metrics"]["loss"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[{experiment_name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Training produced no checkpoint for {experiment_name}.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "experiment_name": experiment_name,
            "model_family": model_family,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "state_dict": copy.deepcopy(best_state),
        },
        checkpoint_path,
    )

    val_eval = evaluate_model(model, val_loader, criterion, device, runtime, class_names)
    test_eval = evaluate_model(model, test_loader, criterion, device, runtime, class_names)

    return {
        "experiment_name": experiment_name,
        "model_family": model_family,
        "device": runtime["device_type"],
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "parameter_count": count_trainable_parameters(model),
        "history": history,
        "val_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "val_bundle": val_eval["bundle"],
        "test_bundle": test_eval["bundle"],
        "output_shape": test_eval["output_shape"],
        "checkpoint_path": str(checkpoint_path),
        "model": model,
    }


def extract_features(model: nn.Module, loader: DataLoader[Any], device: torch.device, runtime: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "extract_features"):
        raise ValueError("Model does not implement extract_features.")

    model.eval()
    features: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images, labels = _move_batch(batch, device, runtime["non_blocking"])
            with _autocast_context(runtime["device_type"], runtime["use_amp"]):
                batch_features = model.extract_features(images)
            features.append(batch_features.float().detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels_all, axis=0)
