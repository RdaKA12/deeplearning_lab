from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    preprocessing: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    max_epochs: int
    patience: int
    min_delta: float
    l2_lambda: float = 0.0

    @property
    def architecture(self) -> str:
        layer_dims = [4, *self.hidden_layers, 1]
        return "-".join(str(dim) for dim in layer_dims)


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    data_path: Path = Path("BankNote_Authentication.csv")
    target_column: str = "class"
    test_size: float = 0.2
    validation_size_within_train: float = 0.2
    output_dir: Path = Path("results")
    threshold: float = 0.5
    learning_rate: float = 0.1
    max_epochs: int = 600
    patience: int = 30
    min_delta: float = 1e-4
    l2_lambda: float = 1e-3
    baseline_hidden_layers: tuple[int, ...] = (6,)
    wider_hidden_layers: tuple[int, ...] = (12,)
    deeper_hidden_layers: tuple[int, ...] = (8, 4)
    feature_columns: tuple[str, ...] = ("variance", "skewness", "curtosis", "entropy")
    experiment_histories_dir: Path = field(init=False)
    experiment_plots_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "experiment_histories_dir", self.output_dir / "histories")
        object.__setattr__(self, "experiment_plots_dir", self.output_dir / "plots")


def build_project_config() -> ProjectConfig:
    return ProjectConfig()
