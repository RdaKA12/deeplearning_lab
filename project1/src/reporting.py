from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_output_dirs(base_dir: Path) -> None:
    (base_dir / "plots").mkdir(parents=True, exist_ok=True)
    (base_dir / "histories").mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def plot_confusion_matrix(matrix: list[list[int]], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(np.array(matrix), cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    scratch_history: list[dict[str, float]],
    torch_history: list[dict[str, float]],
    output_path: Path,
    metric_name: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    scratch_epochs = [row["epoch"] for row in scratch_history]
    torch_epochs = [row["epoch"] for row in torch_history]
    ax.plot(scratch_epochs, [row[f"train_{metric_name}"] for row in scratch_history], label=f"Scratch Train {metric_name.title()}")
    ax.plot(scratch_epochs, [row[f"val_{metric_name}"] for row in scratch_history], label=f"Scratch Val {metric_name.title()}")
    ax.plot(torch_epochs, [row[f"train_{metric_name}"] for row in torch_history], linestyle="--", label=f"PyTorch Train {metric_name.title()}")
    ax.plot(torch_epochs, [row[f"val_{metric_name}"] for row in torch_history], linestyle="--", label=f"PyTorch Val {metric_name.title()}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.title())
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_readme(output_path: Path, content: str) -> None:
    output_path.write_text(content, encoding="utf-8")
