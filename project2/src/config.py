from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    model_key: str
    model_family: str
    max_epochs: int


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 4
    min_delta: float = 1e-4
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "results"
    notebook_path: Path = PROJECT_ROOT / "CIFAR_CNN_Classification.ipynb"
    assignment_pdf_path: Path = PROJECT_ROOT / "yzm304_proje2_2526.pdf"
    docs_gpu_setup_path: Path = REPO_ROOT / "docs" / "windows_gpu_setup.md"
    train_size: int = 45_000
    val_size: int = 5_000
    test_size: int = 10_000
    cuda_num_workers: int = 2
    cpu_num_workers: int = 0
    normalize_mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    normalize_std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    class_names: tuple[str, ...] = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    experiments: tuple[ExperimentDefinition, ...] = (
        ExperimentDefinition("lenet_baseline", "lenet_baseline", "custom_cnn", 20),
        ExperimentDefinition("lenet_bn_dropout", "lenet_bn_dropout", "custom_cnn", 20),
        ExperimentDefinition("resnet18_cifar", "resnet18_cifar", "torchvision_cnn", 18),
    )
    checkpoints_dir: Path = field(init=False)
    features_dir: Path = field(init=False)
    histories_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoints_dir", self.output_dir / "checkpoints")
        object.__setattr__(self, "features_dir", self.output_dir / "features")
        object.__setattr__(self, "histories_dir", self.output_dir / "histories")
        object.__setattr__(self, "plots_dir", self.output_dir / "plots")


def build_project_config() -> ProjectConfig:
    return ProjectConfig()
