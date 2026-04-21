from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_output_dirs(base_dir: Path) -> None:
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "features").mkdir(parents=True, exist_ok=True)
    (base_dir / "histories").mkdir(parents=True, exist_ok=True)
    (base_dir / "plots").mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def plot_training_curves(histories: dict[str, list[dict[str, float]]], output_path: Path, metric_name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for experiment_name, history in histories.items():
        epochs = [int(row["epoch"]) for row in history]
        ax.plot(epochs, [row[f"train_{metric_name}"] for row in history], label=f"{experiment_name} train")
        ax.plot(epochs, [row[f"val_{metric_name}"] for row in history], linestyle="--", label=f"{experiment_name} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[int]], class_names: tuple[str, ...], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(per_class_scores: dict[str, dict[str, float]], class_names: tuple[str, ...], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for experiment_name, class_to_f1 in per_class_scores.items():
        for class_name in class_names:
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "class_name": class_name,
                    "f1": class_to_f1[class_name],
                }
            )

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="class_name", y="f1", hue="experiment_name", ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 comparison")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [title for title, _ in columns]
    separator = ["---"] * len(columns)
    body: list[str] = []
    for row in rows:
        values = []
        for _, key in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |", *body])


def render_readme(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
