from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import ProjectConfig, build_project_config
from src.data_utils import build_data_bundle, build_data_summary, build_runtime_config, set_global_seed
from src.metrics_utils import build_evaluation_bundle
from src.models import build_model
from src.reporting import (
    ensure_output_dirs,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_training_curves,
    render_markdown_table,
    render_readme,
    save_csv,
    save_history,
    save_json,
)
from src.training import extract_features, train_model


def rank_custom_result(result: dict[str, Any]) -> tuple[float, float]:
    return (-result["val_metrics"]["loss"], result["val_metrics"]["accuracy"])


def build_results_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "experiment_name": result["experiment_name"],
                "model_family": result["model_family"],
                "device": result["device"],
                "epochs_trained": result["epochs_trained"],
                "best_epoch": result["best_epoch"],
                "parameter_count": result["parameter_count"],
                "val_accuracy": result["val_metrics"]["accuracy"],
                "val_precision": result["val_metrics"]["precision"],
                "val_recall": result["val_metrics"]["recall"],
                "val_f1": result["val_metrics"]["f1"],
                "val_loss": result["val_metrics"]["loss"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_precision": result["test_metrics"]["precision"],
                "test_recall": result["test_metrics"]["recall"],
                "test_f1": result["test_metrics"]["f1"],
                "test_loss": result["test_metrics"]["loss"],
                "checkpoint_path": result.get("checkpoint_path", ""),
            }
        )
    return rows


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": result["experiment_name"],
        "model_family": result["model_family"],
        "device": result["device"],
        "epochs_trained": result["epochs_trained"],
        "best_epoch": result["best_epoch"],
        "parameter_count": result["parameter_count"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "checkpoint_path": result.get("checkpoint_path", ""),
        "test_bundle": result["test_bundle"],
    }


def build_runtime_summary(config: ProjectConfig, runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_root": str(config.output_dir.parent),
        "device": runtime["device_type"],
        "device_name": runtime["device_name"],
        "cuda_available": runtime["cuda_available"],
        "torch_version": runtime["torch_version"],
        "torch_cuda_version": runtime["torch_cuda_version"],
        "use_amp": runtime["use_amp"],
        "num_workers": runtime["num_workers"],
        "pin_memory": runtime["pin_memory"],
        "cudnn_enabled": runtime["cudnn_enabled"],
    }


def build_readme(
    config: ProjectConfig,
    runtime_summary: dict[str, Any],
    data_summary: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    best_custom: dict[str, Any],
    resnet_result: dict[str, Any],
    hybrid_result: dict[str, Any],
    feature_shapes: dict[str, list[int]],
) -> str:
    baseline_row = next(row for row in summary_rows if row["experiment_name"] == "lenet_baseline")
    best_custom_row = next(row for row in summary_rows if row["experiment_name"] == best_custom["experiment_name"])
    resnet_row = next(row for row in summary_rows if row["experiment_name"] == "resnet18_cifar")
    hybrid_row = next(row for row in summary_rows if row["experiment_name"] == "hybrid_resnet18_svm")
    best_custom_plot = f"results/plots/{best_custom['experiment_name']}_confusion_matrix.png"
    table = render_markdown_table(
        summary_rows,
        [
            ("Experiment", "experiment_name"),
            ("Family", "model_family"),
            ("Device", "device"),
            ("Epochs", "epochs_trained"),
            ("Best Epoch", "best_epoch"),
            ("Val Acc", "val_accuracy"),
            ("Val F1", "val_f1"),
            ("Test Acc", "test_accuracy"),
            ("Test F1", "test_f1"),
            ("Test Loss", "test_loss"),
        ],
    )

    best_custom_report = best_custom["test_bundle"]["classification_report"]
    hybrid_report = hybrid_result["test_bundle"]["classification_report"]
    custom_gain = best_custom_row["test_accuracy"] - baseline_row["test_accuracy"]
    resnet_gain = resnet_row["test_accuracy"] - best_custom_row["test_accuracy"]
    hybrid_gain = hybrid_row["test_accuracy"] - resnet_row["test_accuracy"]
    cat_f1_custom = best_custom["test_bundle"]["per_class_f1"]["cat"]
    cat_f1_resnet = resnet_result["test_bundle"]["per_class_f1"]["cat"]
    cat_f1_hybrid = hybrid_result["test_bundle"]["per_class_f1"]["cat"]
    vehicle_f1_hybrid = {
        "automobile": hybrid_result["test_bundle"]["per_class_f1"]["automobile"],
        "ship": hybrid_result["test_bundle"]["per_class_f1"]["ship"],
        "truck": hybrid_result["test_bundle"]["per_class_f1"]["truck"],
    }
    discussion_lines = [
        f"`{best_custom['experiment_name']}`, `lenet_baseline` modeline gore test accuracy tarafinda `{custom_gain:+.4f}` puan kazanc uretmistir. Bu fark, batch normalization ile aktivasyon dagiliminin daha kararlı hale gelmesi ve dropout ile classifier tarafinda asiri uyumun azalmasi ile aciklanabilir.",
        f"`resnet18_cifar`, en iyi custom modele gore `{resnet_gain:+.4f}` puan ek accuracy kazanci saglamistir. Bunun temel nedeni residual baglantilarin daha derin ozellik hiyerarsisi ogrenmesini kolaylastirmasi ve CIFAR-10 gibi renkli/veri karmasikligi daha yuksek veri setlerinde LeNet ailesine gore daha guclu temsiller uretmesidir.",
        f"Hibrit `LinearSVC`, ResNet embeddingleri uzerinde dogrudan softmax classifier yerine marjin tabanli ayrim yaptiginda `{hybrid_gain:+.4f}` puan ek accuracy elde etti. Bu, `resnet18_cifar` tarafindan uretilen 512-boyutlu ozellik uzayinin lineer olarak daha iyi ayrisabilir oldugunu gosterir.",
        f"Sinif bazinda en zor sinif `cat` oldu: best custom CNN F1 `{cat_f1_custom:.4f}`, ResNet F1 `{cat_f1_resnet:.4f}`, hibrit model F1 `{cat_f1_hybrid:.4f}`. Benzer gorunumlu hayvan siniflari arasindaki karisiklik, confusion matrix'lerde acikca goruluyor.",
        f"Hibrit modelde tasit siniflari daha kararlı ayrildi: automobile F1 `{vehicle_f1_hybrid['automobile']:.4f}`, ship F1 `{vehicle_f1_hybrid['ship']:.4f}`, truck F1 `{vehicle_f1_hybrid['truck']:.4f}`. Kenar/renk/yapisal ipuclari daha ayirt edici oldugu icin bu siniflar hayvan siniflarina gore daha yuksek performans verdi.",
    ]

    return f"""# CIFAR-10 CNN Classification and Hybrid Learning

## Introduction
Bu proje, YZM304 Derin Ogrenme dersi ikinci proje odevini CIFAR-10 veri seti uzerinde tekrar uretilebilir bir deney duzeni ile tamamlamak icin hazirlandi. Hedef, ayni train/validation/test ayrimi uzerinde iki acik yazilmis CNN sinifi, bir literatur tabanli CNN mimarisi ve bir hibrit CNN+SVM yaklasimini egitip karsilastirmaktir.

CIFAR-10 secildi cunku RGB goruntu yapisi LeNet benzeri custom CNN mimarilerini, ResNet tabanli daha guclu bir mimariyi ve bu mimariden ozellik cikarip klasik makine ogrenmesi modeli kullanma senaryosunu ayni problem uzerinde karsilastirmaya izin verir.

GPU kurulumu ve reusable ortam dokumani icin [../docs/windows_gpu_setup.md](../docs/windows_gpu_setup.md) dosyasina bakin.

## Methods
### Dataset and split
- Veri seti: `CIFAR-10`
- Sinif sayisi: `{len(config.class_names)}`
- Siniflar: `{", ".join(config.class_names)}`
- Veri bolmesi: `{data_summary['split_sizes']['train']} train / {data_summary['split_sizes']['val']} val / {data_summary['split_sizes']['test']} test`
- Giris sekli: `{data_summary['sample_shape']}`
- Normalize mean: `{data_summary['normalization']['mean']}`
- Normalize std: `{data_summary['normalization']['std']}`
- Batch size: `{data_summary['batch_size']}`
- Seed: `{config.seed}`

### Theoretical background
Evrişimli sinir aglari, goruntuler uzerinde yerel alici alanlar ve paylasilan agirliklar kullanarak uzamsal desenleri ogrenmeye uygundur. LeNet benzeri yapilar, dusuk seviyeli kenar/doku ozelliklerinden daha soyut siniflandirici ozelliklere giden klasik bir ozellik hiyerarsisi kurar. Batch normalization, ara katman aktivasyonlarini normalize ederek optimizasyonu kararlilastirir; dropout ise ozellikle tam bagli katmanlarda birlikte ezberleme davranisini azaltarak genellemeyi iyilestirmeyi hedefler.

ResNet mimarisindeki residual baglantilar, daha derin aglarda gradyan akisini koruyarak egitimi kolaylastirir. Bu nedenle ResNet18, CIFAR-10 gibi renkli ve sinif ici degiskenligi daha yuksek veri setlerinde LeNet ailesine gore daha guclu temsil ogrenebilir. Hibrit modelde ise son siniflandirici yerine CNN embeddingleri cikarilip lineer SVM ile egitim yapilarak, ogrenilen ozellik uzayinin klasik makine ogrenmesi acisindan ne kadar ayrisabilir oldugu test edilir.

### Model configuration
- `lenet_baseline`: `Conv(3,6,5) -> ReLU -> MaxPool -> Conv(6,16,5) -> ReLU -> MaxPool -> FC(400,120) -> FC(120,84) -> FC(84,10)`
- `lenet_bn_dropout`: temel mimari ile ayni conv/fc boyutlari, ek olarak `BatchNorm2d` ve classifier tarafinda `Dropout(p=0.3)`
- `resnet18_cifar`: `torchvision.models.resnet18(weights=None)` tabanli, `conv1=3x3`, `maxpool=Identity`, `fc=Linear(512,10)`
- Hibrit model: egitilmis `resnet18_cifar` embedding + `Pipeline([StandardScaler(), LinearSVC(C=1.0, max_iter=5000)])`

Tum CNN'ler icin ortak egitim ayarlari:
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam(lr={config.learning_rate}, weight_decay={config.weight_decay})`
- Early stopping patience: `{config.patience}`
- Minimum iyilesme esigi: `{config.min_delta}`
- Epoch plani: `20 / 20 / 18`

### Hyperparameter rationale
- `batch_size=128`: RTX 4070 Laptop GPU uzerinde bellek kullanimi ile kararlı batch istatistikleri arasinda dengeli bir secim saglar.
- `Adam(lr=0.001)`: CIFAR-10 uzerinde sifirdan egitilen CNN'ler icin hizli ama kontrol edilebilir yakinlama sagladigi icin secildi.
- `weight_decay=1e-4`: ozellikle custom CNN'lerde genel performansi korurken asiri uyumu sinirlamak icin hafif bir L2 etkisi yaratir.
- `patience=4`: validation loss plato yaptiginda gereksiz epoch harcamadan en iyi checkpoint'i korumayi saglar.
- ResNet icin pretrained agirlik kullanilmadi: karsilastirma ayni veri seti ve ayni egitim senaryosu icinde adil kalsin diye `weights=None` secildi.

Calisma zamani:
- Device: `{runtime_summary['device']}`
- Device name: `{runtime_summary['device_name']}`
- Torch: `{runtime_summary['torch_version']}`
- Torch CUDA: `{runtime_summary['torch_cuda_version']}`
- AMP: `{runtime_summary['use_amp']}`

### Hybrid feature export
`resnet18_cifar` modelinin `fc` oncesi `512` boyutlu embedding'i `results/features/` altina `.npy` olarak yazildi:
- `train_features.npy`: `{feature_shapes['train_features']}`
- `val_features.npy`: `{feature_shapes['val_features']}`
- `test_features.npy`: `{feature_shapes['test_features']}`
- `train_labels.npy`: `{feature_shapes['train_labels']}`
- `val_labels.npy`: `{feature_shapes['val_labels']}`
- `test_labels.npy`: `{feature_shapes['test_labels']}`

### Reproducibility
GPU hazir reusable ortam ile tekrar calistirmak icin:

```bash
cd project2
%USERPROFILE%\\dl-gpu-py313\\Scripts\\python.exe run_experiments.py
```

## Results
### Experiment summary
{table}

### Selected models
- Best custom CNN: `{best_custom['experiment_name']}` with test accuracy `{best_custom['test_metrics']['accuracy']:.4f}` and macro F1 `{best_custom['test_metrics']['f1']:.4f}`
- ResNet18 test accuracy: `{resnet_result['test_metrics']['accuracy']:.4f}`
- Hybrid CNN+SVM test accuracy: `{hybrid_result['test_metrics']['accuracy']:.4f}`

### Additional observations
- Best custom CNN class-level macro avg F1: `{best_custom_report['macro avg']['f1-score']:.4f}`
- Hybrid class-level macro avg F1: `{hybrid_report['macro avg']['f1-score']:.4f}`
- Loss and accuracy curves: `results/plots/loss_curves.png`, `results/plots/accuracy_curves.png`
- Confusion matrices: `{best_custom_plot}`, `results/plots/resnet18_cifar_confusion_matrix.png`, `results/plots/hybrid_resnet18_svm_confusion_matrix.png`
- Per-class F1 comparison: `results/plots/per_class_f1_comparison.png`

### Visuals
![Loss curves](results/plots/loss_curves.png)

![Accuracy curves](results/plots/accuracy_curves.png)

![Best custom CNN confusion matrix]({best_custom_plot})

![ResNet18 confusion matrix](results/plots/resnet18_cifar_confusion_matrix.png)

![Hybrid CNN+SVM confusion matrix](results/plots/hybrid_resnet18_svm_confusion_matrix.png)

![Per-class F1 comparison](results/plots/per_class_f1_comparison.png)

## Discussion
{" ".join(discussion_lines)}

Bu projenin temel sinirliligi, tek veri seti uzerinde ve sabit hiperparametrelerle karsilastirma yapmasidir. Ayrica custom CNN ailesi, ResNet18 kadar derin veya kapsayici degildir; bu nedenle mutlak performans farkinin bir bolumu mimari kapasiteden gelir. Gelecek adimlarda veri artirma, farkli optimizer secenekleri, pretrained agirliklar, daha genis hiperparametre taramasi ve farkli hibrit siniflandiricilar ile karsilastirma genisletilebilir.

## References
- PyTorch Get Started: <https://pytorch.org/get-started/locally/>
- PyTorch 2.11 Release Blog: <https://pytorch.org/blog/pytorch-2-11-release-blog/>
- NVIDIA CUDA Toolkit Release Notes: <https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html>
"""


def run_hybrid_pipeline(
    config: ProjectConfig,
    runtime: dict[str, Any],
    data_bundle: Any,
    resnet_result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    resnet_model = resnet_result["model"]
    device = runtime["device"]

    train_features, train_labels = extract_features(resnet_model, data_bundle.train_eval_loader, device, runtime)
    val_features, val_labels = extract_features(resnet_model, data_bundle.val_loader, device, runtime)
    test_features, test_labels = extract_features(resnet_model, data_bundle.test_loader, device, runtime)

    feature_files = {
        "train_features": config.features_dir / "train_features.npy",
        "val_features": config.features_dir / "val_features.npy",
        "test_features": config.features_dir / "test_features.npy",
        "train_labels": config.features_dir / "train_labels.npy",
        "val_labels": config.features_dir / "val_labels.npy",
        "test_labels": config.features_dir / "test_labels.npy",
    }

    np.save(feature_files["train_features"], train_features)
    np.save(feature_files["val_features"], val_features)
    np.save(feature_files["test_features"], test_features)
    np.save(feature_files["train_labels"], train_labels)
    np.save(feature_files["val_labels"], val_labels)
    np.save(feature_files["test_labels"], test_labels)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=1.0, max_iter=5000)),
        ]
    )
    pipeline.fit(train_features, train_labels)
    val_pred = pipeline.predict(val_features)
    test_pred = pipeline.predict(test_features)

    val_bundle = build_evaluation_bundle(val_labels, val_pred, config.class_names)
    test_bundle = build_evaluation_bundle(test_labels, test_pred, config.class_names)
    hybrid_result = {
        "experiment_name": "hybrid_resnet18_svm",
        "model_family": "hybrid_ml",
        "device": runtime["device_type"],
        "epochs_trained": 0,
        "best_epoch": 0,
        "parameter_count": int(train_features.shape[1]),
        "val_metrics": {
            "loss": 0.0,
            "accuracy": val_bundle["accuracy"],
            "precision": val_bundle["precision"],
            "recall": val_bundle["recall"],
            "f1": val_bundle["f1"],
        },
        "test_metrics": {
            "loss": 0.0,
            "accuracy": test_bundle["accuracy"],
            "precision": test_bundle["precision"],
            "recall": test_bundle["recall"],
            "f1": test_bundle["f1"],
        },
        "test_bundle": test_bundle,
        "feature_files": {key: str(path) for key, path in feature_files.items()},
    }
    feature_shapes = {
        "train_features": list(train_features.shape),
        "val_features": list(val_features.shape),
        "test_features": list(test_features.shape),
        "train_labels": list(train_labels.shape),
        "val_labels": list(val_labels.shape),
        "test_labels": list(test_labels.shape),
    }
    return hybrid_result, feature_shapes


def build_verification(
    config: ProjectConfig,
    data_summary: dict[str, Any],
    runtime_summary: dict[str, Any],
    cnn_results: list[dict[str, Any]],
    best_custom: dict[str, Any],
    hybrid_result: dict[str, Any],
    feature_shapes: dict[str, list[int]],
) -> dict[str, Any]:
    required_plots = [
        config.plots_dir / "loss_curves.png",
        config.plots_dir / "accuracy_curves.png",
        config.plots_dir / f"{best_custom['experiment_name']}_confusion_matrix.png",
        config.plots_dir / "resnet18_cifar_confusion_matrix.png",
        config.plots_dir / "hybrid_resnet18_svm_confusion_matrix.png",
        config.plots_dir / "per_class_f1_comparison.png",
    ]
    history_files = [config.histories_dir / f"{result['experiment_name']}.csv" for result in cnn_results]
    return {
        "runtime_checks": {
            "cuda_available": runtime_summary["cuda_available"],
            "device": runtime_summary["device"],
            "use_amp_matches_device": runtime_summary["use_amp"] == (runtime_summary["device"] == "cuda"),
        },
        "split_checks": {
            "train_size_ok": data_summary["split_sizes"]["train"] == config.train_size,
            "val_size_ok": data_summary["split_sizes"]["val"] == config.val_size,
            "test_size_ok": data_summary["split_sizes"]["test"] == config.test_size,
        },
        "shape_checks": {
            "sample_shape_ok": data_summary["sample_shape"] == [3, 32, 32],
            "cnn_output_shapes": {result["experiment_name"]: result["output_shape"] for result in cnn_results},
            "cnn_output_shape_ok": all(result["output_shape"][1] == 10 for result in cnn_results),
            "feature_shapes": feature_shapes,
            "feature_dimension_ok": feature_shapes["train_features"][1] == 512,
            "feature_label_alignment_ok": feature_shapes["train_features"][0] == feature_shapes["train_labels"][0]
            and feature_shapes["val_features"][0] == feature_shapes["val_labels"][0]
            and feature_shapes["test_features"][0] == feature_shapes["test_labels"][0],
        },
        "training_checks": {
            "history_files_exist": all(path.exists() for path in history_files),
            "plot_files_exist": all(path.exists() for path in required_plots),
            "loss_decreases": {
                result["experiment_name"]: result["history"][-1]["train_loss"] < result["history"][0]["train_loss"]
                for result in cnn_results
            },
            "predictions_not_collapsed": {
                result["experiment_name"]: int((np.array(result["test_bundle"]["confusion_matrix"]).sum(axis=0) > 0).sum()) > 1
                for result in cnn_results
            },
            "summary_contains_all_experiments": len(cnn_results) + 1 == 4 and hybrid_result["experiment_name"] == "hybrid_resnet18_svm",
        },
    }


def main() -> None:
    config = build_project_config()
    set_global_seed(config.seed)
    runtime = build_runtime_config(config)
    ensure_output_dirs(config.output_dir)
    print(f"Running project2 on device={runtime['device_type']} ({runtime['device_name']})")

    data_bundle = build_data_bundle(config, runtime)
    data_summary = build_data_summary(config, data_bundle)
    runtime_summary = build_runtime_summary(config, runtime)
    save_json(config.output_dir / "data_summary.json", data_summary)
    save_json(config.output_dir / "runtime_summary.json", runtime_summary)

    cnn_results: list[dict[str, Any]] = []
    for experiment in config.experiments:
        print(f"Starting experiment: {experiment.name}")
        checkpoint_path = config.checkpoints_dir / f"{experiment.name}.pt"
        result = train_model(
            experiment_name=experiment.name,
            model_family=experiment.model_family,
            model=build_model(experiment.model_key),
            train_loader=data_bundle.train_loader,
            val_loader=data_bundle.val_loader,
            test_loader=data_bundle.test_loader,
            device=runtime["device"],
            runtime=runtime,
            class_names=config.class_names,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_epochs=experiment.max_epochs,
            patience=config.patience,
            min_delta=config.min_delta,
            checkpoint_path=checkpoint_path,
        )
        cnn_results.append(result)
        save_history(config.histories_dir / f"{experiment.name}.csv", result["history"])
        print(f"Finished experiment: {experiment.name}")

    cnn_histories = {result["experiment_name"]: result["history"] for result in cnn_results}
    plot_training_curves(cnn_histories, config.plots_dir / "loss_curves.png", "loss", "Training and validation loss")
    plot_training_curves(cnn_histories, config.plots_dir / "accuracy_curves.png", "accuracy", "Training and validation accuracy")

    best_custom = max((result for result in cnn_results if result["model_family"] == "custom_cnn"), key=rank_custom_result)
    resnet_result = next(result for result in cnn_results if result["experiment_name"] == "resnet18_cifar")
    hybrid_result, feature_shapes = run_hybrid_pipeline(config, runtime, data_bundle, resnet_result)
    print("Finished hybrid pipeline")

    plot_confusion_matrix(
        best_custom["test_bundle"]["confusion_matrix"],
        config.class_names,
        f"{best_custom['experiment_name']} confusion matrix",
        config.plots_dir / f"{best_custom['experiment_name']}_confusion_matrix.png",
    )
    plot_confusion_matrix(
        resnet_result["test_bundle"]["confusion_matrix"],
        config.class_names,
        "resnet18_cifar confusion matrix",
        config.plots_dir / "resnet18_cifar_confusion_matrix.png",
    )
    plot_confusion_matrix(
        hybrid_result["test_bundle"]["confusion_matrix"],
        config.class_names,
        "hybrid_resnet18_svm confusion matrix",
        config.plots_dir / "hybrid_resnet18_svm_confusion_matrix.png",
    )
    plot_per_class_f1(
        {
            best_custom["experiment_name"]: best_custom["test_bundle"]["per_class_f1"],
            "resnet18_cifar": resnet_result["test_bundle"]["per_class_f1"],
            "hybrid_resnet18_svm": hybrid_result["test_bundle"]["per_class_f1"],
        },
        config.class_names,
        config.plots_dir / "per_class_f1_comparison.png",
    )

    all_results = cnn_results + [hybrid_result]
    summary_rows = build_results_table(all_results)
    save_csv(config.output_dir / "experiment_summary.csv", summary_rows)
    save_json(
        config.output_dir / "final_summary.json",
        {
            "runtime": runtime_summary,
            "data_summary": data_summary,
            "best_custom_experiment": best_custom["experiment_name"],
            "experiments": [compact_result(result) for result in all_results],
            "feature_shapes": feature_shapes,
        },
    )
    verification = build_verification(config, data_summary, runtime_summary, cnn_results, best_custom, hybrid_result, feature_shapes)
    save_json(config.output_dir / "verification.json", verification)

    readme_content = build_readme(config, runtime_summary, data_summary, summary_rows, best_custom, resnet_result, hybrid_result, feature_shapes)
    render_readme(config.output_dir.parent / "README.md", readme_content)
    print("Artifacts and README generated.")


if __name__ == "__main__":
    main()
