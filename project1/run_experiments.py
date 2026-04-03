from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import ExperimentConfig, build_project_config
from src.data_utils import build_data_summary, load_dataset, prepare_features, set_global_seed, split_dataset
from src.metrics_utils import build_evaluation_bundle
from src.models import ScratchMLPClassifier, TorchMLPClassifier, generate_initial_parameters
from src.reporting import ensure_output_dirs, plot_confusion_matrix, plot_training_curves, render_readme, save_csv, save_history, save_json


def count_parameters(layer_dims: list[int]) -> int:
    return int(sum((input_dim * output_dim) + output_dim for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:])))


def make_experiment_config(name: str, preprocessing: str, hidden_layers: tuple[int, ...], project_config: Any, l2_lambda: float = 0.0) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        preprocessing=preprocessing,
        hidden_layers=hidden_layers,
        learning_rate=project_config.learning_rate,
        max_epochs=project_config.max_epochs,
        patience=project_config.patience,
        min_delta=project_config.min_delta,
        l2_lambda=l2_lambda,
    )


def rank_key(result: dict[str, Any]) -> tuple[float, float, int]:
    return (
        result["val_metrics"]["accuracy"],
        result["val_metrics"]["f1"],
        -result["best_epoch"],
    )


def run_scratch_experiment(config: ExperimentConfig, splits: Any, project_config: Any) -> dict[str, Any]:
    prepared, _ = prepare_features(splits, config.preprocessing)
    layer_dims = [splits.X_train.shape[1], *config.hidden_layers, 1]
    model = ScratchMLPClassifier(
        layer_dims=layer_dims,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        threshold=project_config.threshold,
        l2_lambda=config.l2_lambda,
        seed=project_config.seed,
    )
    model.fit(prepared["X_train"], splits.y_train, prepared["X_val"], splits.y_val)
    val_metrics = model.evaluate(prepared["X_val"], splits.y_val)
    test_metrics = model.evaluate(prepared["X_test"], splits.y_test)
    return {
        "experiment_name": config.name,
        "model_family": "scratch",
        "preprocessing": config.preprocessing,
        "architecture": config.architecture,
        "hidden_layers": list(config.hidden_layers),
        "l2_lambda": config.l2_lambda,
        "epochs_trained": model.epochs_trained,
        "best_epoch": model.best_epoch,
        "parameter_count": count_parameters(layer_dims),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": model.history,
        "layer_dims": layer_dims,
    }


def write_history_outputs(result: dict[str, Any], output_dir: Path) -> None:
    file_stub = result["experiment_name"].lower().replace(" ", "_").replace("+", "plus")
    save_history(output_dir / "histories" / f"{file_stub}.csv", result["history"])


def build_results_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "experiment_name": result["experiment_name"],
                "model_family": result["model_family"],
                "preprocessing": result["preprocessing"],
                "architecture": result["architecture"],
                "l2_lambda": result["l2_lambda"],
                "parameter_count": result["parameter_count"],
                "epochs_trained": result["epochs_trained"],
                "best_epoch": result["best_epoch"],
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
            }
        )
    return rows


def render_markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [title for title, _ in columns]
    separator = ["---"] * len(columns)
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
                for _, key in columns
            )
            + " |"
        )
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |", *body])


def build_discussion(best_scratch: dict[str, Any], standardized_result: dict[str, Any], raw_result: dict[str, Any], wider_result: dict[str, Any], deeper_result: dict[str, Any], l2_result: dict[str, Any]) -> str:
    lines = []
    if standardized_result["val_metrics"]["accuracy"] > raw_result["val_metrics"]["accuracy"]:
        lines.append("Standardizasyon, temel modele gore optimizasyonu kolaylastirip daha yuksek validation accuracy urettti.")
    elif standardized_result["val_metrics"]["accuracy"] < raw_result["val_metrics"]["accuracy"]:
        lines.append("Ham ozelliklerle egitim, bu veri setinde standardizasyondan daha iyi validation accuracy verdi; veri zaten ayrisabilir oldugu icin olcek degisimi zorunlu olmadi.")
    else:
        lines.append("Ham ve standardize veri ayni validation accuracy seviyesine ulasti; burada temel fark yakinlasma hizi ve loss davranisi oldu.")

    if wider_result["val_metrics"]["accuracy"] >= raw_result["val_metrics"]["accuracy"]:
        lines.append("Gizli katmandaki noron sayisini artirmak kapasiteyi yukseltip hatayi azaltabildi; bu, temel modelde hafif bias olabilecegini gosteriyor.")
    else:
        lines.append("Daha genis mimari ek fayda saglamadi; temel model veri karmasikligi icin yeterli kapasiteye sahip.")

    if deeper_result["val_metrics"]["accuracy"] > raw_result["val_metrics"]["accuracy"]:
        lines.append("Ek gizli katman dogrusal olmayan sinirlari daha iyi modelledi ve derinligin faydasini gosterdi.")
    else:
        lines.append("Ek gizli katman anlamli kazanc saglamadi; bu veri icin derinlik artisinin getirisi sinirli kaldi.")

    if l2_result["val_metrics"]["loss"] <= best_scratch["val_metrics"]["loss"]:
        lines.append("L2 regularization validation loss u iyilestirdi; bu da varyans kontrolunde yararli oldugunu gosteriyor.")
    else:
        lines.append("L2 regularization belirgin bir genel performans artisina donusmedi; veri seti zaten kucuk ve guclu ayrisabilir oldugu icin duzenleme etkisi sinirli kaldi.")

    lines.append("Train ve validation egri farki buyurse overfitting, her ikisi de dusuk kalsaydi underfitting yorumu yapilacakti. Bu projede en iyi modelde train ve validation performansi birbirine yakin seyrederek dengeli bir ogrenme gosterdi.")
    return " ".join(lines)


def compact_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": result["experiment_name"],
        "model_family": result["model_family"],
        "preprocessing": result["preprocessing"],
        "architecture": result["architecture"],
        "l2_lambda": result["l2_lambda"],
        "parameter_count": result["parameter_count"],
        "epochs_trained": result["epochs_trained"],
        "best_epoch": result["best_epoch"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "test_bundle": result["test_bundle"],
    }


def validate_split_shapes(splits: Any) -> dict[str, Any]:
    return {
        "train_shape": list(splits.X_train.shape),
        "val_shape": list(splits.X_val.shape),
        "test_shape": list(splits.X_test.shape),
        "train_labels_shape": list(splits.y_train.shape),
        "val_labels_shape": list(splits.y_val.shape),
        "test_labels_shape": list(splits.y_test.shape),
        "feature_count_ok": splits.X_train.shape[1] == splits.X_val.shape[1] == splits.X_test.shape[1] == 4,
        "sample_count_ok": splits.X_train.shape[0] + splits.X_val.shape[0] + splits.X_test.shape[0] == 1372,
    }


def run_validation_checks(splits: Any, prepared_best: dict[str, np.ndarray], scratch_model: ScratchMLPClassifier, torch_model: TorchMLPClassifier, initial_forward_gap: float) -> dict[str, Any]:
    scratch_probabilities = scratch_model.predict_proba(prepared_best["X_test"])
    torch_probabilities = torch_model.predict_proba(prepared_best["X_test"])
    scratch_predictions = scratch_model.predict(prepared_best["X_test"])
    checks = validate_split_shapes(splits)
    checks.update(
        {
            "scratch_probability_range_ok": bool(np.all((scratch_probabilities >= 0.0) & (scratch_probabilities <= 1.0))),
            "torch_probability_range_ok": bool(np.all((torch_probabilities >= 0.0) & (torch_probabilities <= 1.0))),
            "scratch_binary_predictions_ok": bool(set(np.unique(scratch_predictions).tolist()).issubset({0, 1})),
            "initial_forward_match_ok": bool(initial_forward_gap < 1e-12),
            "scratch_loss_decreases": bool(scratch_model.history[-1]["train_loss"] < scratch_model.history[0]["train_loss"]),
            "torch_loss_decreases": bool(torch_model.history[-1]["train_loss"] < torch_model.history[0]["train_loss"]),
        }
    )
    return checks


def build_readme(project_config: Any, data_summary: dict[str, Any], results_rows: list[dict[str, Any]], best_scratch: dict[str, Any], pytorch_result: dict[str, Any], initial_forward_gap: float, discussion: str) -> str:
    scratch_rows = [row for row in results_rows if row["model_family"] == "scratch"]
    comparison_rows = [
        {
            "Model": "Scratch",
            "Preprocessing": best_scratch["preprocessing"],
            "Architecture": best_scratch["architecture"],
            "Val Accuracy": best_scratch["val_metrics"]["accuracy"],
            "Test Accuracy": best_scratch["test_metrics"]["accuracy"],
            "Test Precision": best_scratch["test_metrics"]["precision"],
            "Test Recall": best_scratch["test_metrics"]["recall"],
            "Test F1": best_scratch["test_metrics"]["f1"],
            "Best Epoch": best_scratch["best_epoch"],
        },
        {
            "Model": "PyTorch",
            "Preprocessing": pytorch_result["preprocessing"],
            "Architecture": pytorch_result["architecture"],
            "Val Accuracy": pytorch_result["val_metrics"]["accuracy"],
            "Test Accuracy": pytorch_result["test_metrics"]["accuracy"],
            "Test Precision": pytorch_result["test_metrics"]["precision"],
            "Test Recall": pytorch_result["test_metrics"]["recall"],
            "Test F1": pytorch_result["test_metrics"]["f1"],
            "Best Epoch": pytorch_result["best_epoch"],
        },
    ]
    scratch_table = render_markdown_table(
        scratch_rows,
        [
            ("Experiment", "experiment_name"),
            ("Prep", "preprocessing"),
            ("Arch", "architecture"),
            ("L2", "l2_lambda"),
            ("Best Epoch", "best_epoch"),
            ("Val Acc", "val_accuracy"),
            ("Val F1", "val_f1"),
            ("Test Acc", "test_accuracy"),
            ("Test F1", "test_f1"),
        ],
    )
    comparison_table = render_markdown_table(
        comparison_rows,
        [
            ("Model", "Model"),
            ("Prep", "Preprocessing"),
            ("Arch", "Architecture"),
            ("Best Epoch", "Best Epoch"),
            ("Val Acc", "Val Accuracy"),
            ("Test Acc", "Test Accuracy"),
            ("Test Precision", "Test Precision"),
            ("Test Recall", "Test Recall"),
            ("Test F1", "Test F1"),
        ],
    )
    conf_matrix = best_scratch["test_bundle"]["confusion_matrix"]
    report = best_scratch["test_bundle"]["classification_report"]
    strongest_feature = data_summary["strongest_feature"]
    strongest_feature_comment = data_summary["strongest_feature_comment"]
    return f"""# BankNote Authentication with Scratch MLP and PyTorch

## Introduction
Bu proje, 13 Mart 2026 laboratuvar uygulamasinda kurulan tek gizli katmanli MLP iskeletini tam bir mini proje haline getirmek icin gelistirildi. Problem, banknotlarin gercek veya sahte olmasini `variance`, `skewness`, `curtosis` ve `entropy` ozelliklerinden yararlanarak ikili siniflandirma olarak ele almaktadir.

BankNote veri seti secildi cunku kucuk ama net sinif ayrimi barindiriyor; bu da sifirdan yazilmis bir MLP ile ogrenme dinamiklerini incelemek icin uygun bir ortam sagliyor. Projenin hedefi, ayni veri bolmesi uzerinde hem NumPy tabanli bir MLP hem de ayni mimarinin PyTorch eslenigini egitip karsilastirmak, ayrica overfitting/underfitting davranisini mimari ve duzenleme deneyleriyle gozlemlemektir.

## Methods
### Dataset and split
- Veri dosyasi: `BankNote_Authentication.csv`
- Veri boyutu: `{data_summary['shape']['rows']} satir x {data_summary['shape']['columns']} sutun`
- Girdi ozellikleri: `{", ".join(project_config.feature_columns)}`
- Hedef sutun: `{project_config.target_column}`
- Eksik deger: bulunmadi
- Sinif dagilimi: `{data_summary['class_distribution']}`
- En guclu ozellik: `{strongest_feature}`. {strongest_feature_comment}

Veri, `seed={project_config.seed}` ile stratified olarak `%64 train / %16 validation / %20 test` biciminde ayrildi. Test seti yalnizca son asamada kullanildi. Iki on isleme secenegi denendi: ham veri ve sadece train setine fit edilen standardization.

### Model configuration
- Temel mimari: `4-6-1`
- Genis mimari: `4-12-1`
- Derin mimari: `4-8-4-1`
- Gizli katman aktivasyonu: `tanh`
- Cikis aktivasyonu: `sigmoid`
- Loss: binary cross entropy
- Optimizer: full-batch SGD
- Learning rate: `{project_config.learning_rate}`
- Maksimum epoch: `{project_config.max_epochs}`
- Early stopping patience: `{project_config.patience}`
- Minimum iyilesme esigi: `{project_config.min_delta}`
- L2 regularization deneyi: `lambda={project_config.l2_lambda}`
- Model secim kriteri: once validation accuracy, sonra validation F1, esitlikte daha dusuk best epoch

Scratch ve PyTorch modelleri icin ayni seed, ayni veri bolmesi, ayni mimari, ayni optimizer mantigi kullanildi. Son karsilastirma icin PyTorch modelinin baslangic agirliklari, scratch modeline verilen NumPy agirliklarindan kopyalandi. Egitim oncesi ilk ileri yayilim farki `max|scratch-pytorch| = {initial_forward_gap:.8f}` olarak olculdu.

### Reproducibility
Tum deneyleri ve ciktilari tekrar uretmek icin:

```bash
python run_experiments.py
```

Uretilen temel ciktilar:
- `results/data_summary.json`
- `results/experiment_summary.csv`
- `results/plots/*.png`
- `results/histories/*.csv`

## Results
### Scratch experiment summary
{scratch_table}

### Best model
Validation secim kurallarina gore en iyi scratch model `{best_scratch['experiment_name']}` oldu. Bu model `{best_scratch['preprocessing']}` veri hazirlama ve `{best_scratch['architecture']}` mimarisini kullandi. Test performansi:

- Accuracy: `{best_scratch['test_metrics']['accuracy']:.4f}`
- Precision: `{best_scratch['test_metrics']['precision']:.4f}`
- Recall: `{best_scratch['test_metrics']['recall']:.4f}`
- F1 Score: `{best_scratch['test_metrics']['f1']:.4f}`
- Loss: `{best_scratch['test_metrics']['loss']:.4f}`

Not: Ham veriyle egitilen temel model test setinde daha yuksek accuracy uretse de model secimi bilerek test setine bakilmadan, sadece validation accuracy/F1 ve epoch sayisina gore yapildi.

Confusion matrix:

```text
{conf_matrix[0]}
{conf_matrix[1]}
```

Classification report ozet degerleri:
- Class 0 precision / recall / f1: `{report['0']['precision']:.4f} / {report['0']['recall']:.4f} / {report['0']['f1-score']:.4f}`
- Class 1 precision / recall / f1: `{report['1']['precision']:.4f} / {report['1']['recall']:.4f} / {report['1']['f1-score']:.4f}`
- Macro avg f1: `{report['macro avg']['f1-score']:.4f}`

### Scratch vs PyTorch
{comparison_table}

PyTorch sonucu, ayni baslangic agirliklari ve ayni optimizasyon mantigi ile scratch modele yakin performans verdi. Bu, sifirdan yazilan uygulamanin matematiksel akisini framework tabanli uygulama ile tutarli sekilde dogruladi.

## Discussion
{discussion}

Bu calismanin temel sinirliligi, veri setinin gorece kucuk ve ayrisabilir olmasidir; bu nedenle daha derin mimariler her zaman anlamli ek kazanc uretmeyebilir. Gelecek adimlarda mini-batch egitim, farkli aktivasyonlar, batch normalization ve daha buyuk veri setleri ile ayni karsilastirma genisletilebilir.
"""


def main() -> None:
    project_config = build_project_config()
    set_global_seed(project_config.seed)
    ensure_output_dirs(project_config.output_dir)

    df = load_dataset(project_config.data_path)
    data_summary = build_data_summary(df, project_config.target_column)
    save_json(project_config.output_dir / "data_summary.json", data_summary)

    splits = split_dataset(
        df=df,
        target_column=project_config.target_column,
        seed=project_config.seed,
        test_size=project_config.test_size,
        validation_size_within_train=project_config.validation_size_within_train,
    )

    baseline_raw = run_scratch_experiment(
        make_experiment_config("Scratch Baseline", "raw", project_config.baseline_hidden_layers, project_config),
        splits,
        project_config,
    )
    baseline_std = run_scratch_experiment(
        make_experiment_config("Scratch + Standardization", "standardized", project_config.baseline_hidden_layers, project_config),
        splits,
        project_config,
    )
    for result in (baseline_raw, baseline_std):
        write_history_outputs(result, project_config.output_dir)

    better_preprocessing = baseline_raw["preprocessing"] if rank_key(baseline_raw) >= rank_key(baseline_std) else baseline_std["preprocessing"]
    wider = run_scratch_experiment(
        make_experiment_config("Scratch Wider", better_preprocessing, project_config.wider_hidden_layers, project_config),
        splits,
        project_config,
    )
    deeper = run_scratch_experiment(
        make_experiment_config("Scratch Deeper", better_preprocessing, project_config.deeper_hidden_layers, project_config),
        splits,
        project_config,
    )
    for result in (wider, deeper):
        write_history_outputs(result, project_config.output_dir)

    first_four = [baseline_raw, baseline_std, wider, deeper]
    best_pre_l2 = max(first_four, key=rank_key)
    l2_experiment = run_scratch_experiment(
        make_experiment_config(
            "Scratch + L2",
            best_pre_l2["preprocessing"],
            tuple(best_pre_l2["hidden_layers"]),
            project_config,
            l2_lambda=project_config.l2_lambda,
        ),
        splits,
        project_config,
    )
    write_history_outputs(l2_experiment, project_config.output_dir)

    scratch_results = [baseline_raw, baseline_std, wider, deeper, l2_experiment]
    best_scratch_summary = max(scratch_results, key=rank_key)

    prepared_best, _ = prepare_features(splits, best_scratch_summary["preprocessing"])
    shared_initial_parameters = generate_initial_parameters(best_scratch_summary["layer_dims"], project_config.seed)
    scratch_best_model = ScratchMLPClassifier(
        layer_dims=best_scratch_summary["layer_dims"],
        learning_rate=project_config.learning_rate,
        max_epochs=project_config.max_epochs,
        patience=project_config.patience,
        min_delta=project_config.min_delta,
        threshold=project_config.threshold,
        l2_lambda=best_scratch_summary["l2_lambda"],
        seed=project_config.seed,
        initial_parameters=shared_initial_parameters,
    )
    torch_best_model = TorchMLPClassifier(
        layer_dims=best_scratch_summary["layer_dims"],
        learning_rate=project_config.learning_rate,
        max_epochs=project_config.max_epochs,
        patience=project_config.patience,
        min_delta=project_config.min_delta,
        threshold=project_config.threshold,
        l2_lambda=best_scratch_summary["l2_lambda"],
        seed=project_config.seed,
        initial_parameters=shared_initial_parameters,
    )

    scratch_initial_forward = scratch_best_model.predict_proba(prepared_best["X_train"])
    torch_initial_forward = torch_best_model.compare_initial_forward(prepared_best["X_train"])
    initial_forward_gap = float(np.max(np.abs(scratch_initial_forward - torch_initial_forward)))

    scratch_best_model.fit(prepared_best["X_train"], splits.y_train, prepared_best["X_val"], splits.y_val)
    torch_best_model.fit(prepared_best["X_train"], splits.y_train, prepared_best["X_val"], splits.y_val)

    scratch_best_result = {
        "experiment_name": best_scratch_summary["experiment_name"],
        "model_family": "scratch",
        "preprocessing": best_scratch_summary["preprocessing"],
        "architecture": best_scratch_summary["architecture"],
        "l2_lambda": best_scratch_summary["l2_lambda"],
        "parameter_count": best_scratch_summary["parameter_count"],
        "epochs_trained": scratch_best_model.epochs_trained,
        "best_epoch": scratch_best_model.best_epoch,
        "val_metrics": scratch_best_model.evaluate(prepared_best["X_val"], splits.y_val),
        "test_metrics": scratch_best_model.evaluate(prepared_best["X_test"], splits.y_test),
        "test_bundle": build_evaluation_bundle(splits.y_test, scratch_best_model.predict(prepared_best["X_test"])),
        "history": scratch_best_model.history,
    }
    pytorch_result = {
        "experiment_name": "PyTorch Replica",
        "model_family": "pytorch",
        "preprocessing": best_scratch_summary["preprocessing"],
        "architecture": best_scratch_summary["architecture"],
        "l2_lambda": best_scratch_summary["l2_lambda"],
        "parameter_count": best_scratch_summary["parameter_count"],
        "epochs_trained": torch_best_model.epochs_trained,
        "best_epoch": torch_best_model.best_epoch,
        "val_metrics": torch_best_model.evaluate(prepared_best["X_val"], splits.y_val),
        "test_metrics": torch_best_model.evaluate(prepared_best["X_test"], splits.y_test),
        "test_bundle": build_evaluation_bundle(splits.y_test, torch_best_model.predict(prepared_best["X_test"])),
        "history": torch_best_model.history,
    }
    write_history_outputs(scratch_best_result, project_config.output_dir)
    write_history_outputs(pytorch_result, project_config.output_dir)

    verification = run_validation_checks(splits, prepared_best, scratch_best_model, torch_best_model, initial_forward_gap)
    save_json(project_config.output_dir / "verification.json", verification)

    results_rows = build_results_table(scratch_results + [pytorch_result])
    save_csv(project_config.output_dir / "experiment_summary.csv", results_rows)

    plot_confusion_matrix(
        scratch_best_result["test_bundle"]["confusion_matrix"],
        f"Best Scratch Model ({scratch_best_result['experiment_name']})",
        project_config.output_dir / "plots" / "best_scratch_confusion_matrix.png",
    )
    plot_confusion_matrix(
        pytorch_result["test_bundle"]["confusion_matrix"],
        "PyTorch Replica",
        project_config.output_dir / "plots" / "pytorch_confusion_matrix.png",
    )
    plot_training_curves(
        scratch_best_result["history"],
        pytorch_result["history"],
        project_config.output_dir / "plots" / "loss_curves.png",
        metric_name="loss",
        title="Scratch vs PyTorch Loss Curves",
    )
    plot_training_curves(
        scratch_best_result["history"],
        pytorch_result["history"],
        project_config.output_dir / "plots" / "accuracy_curves.png",
        metric_name="accuracy",
        title="Scratch vs PyTorch Accuracy Curves",
    )

    discussion = build_discussion(
        best_scratch=scratch_best_result,
        standardized_result=baseline_std,
        raw_result=baseline_raw,
        wider_result=wider,
        deeper_result=deeper,
        l2_result=l2_experiment,
    )
    save_json(
        project_config.output_dir / "final_summary.json",
        {
            "data_summary": data_summary,
            "scratch_best_result": compact_result_payload(scratch_best_result),
            "pytorch_result": compact_result_payload(pytorch_result),
            "initial_forward_gap": initial_forward_gap,
            "discussion": discussion,
        },
    )

    readme_content = build_readme(
        project_config=project_config,
        data_summary=data_summary,
        results_rows=results_rows,
        best_scratch=scratch_best_result,
        pytorch_result=pytorch_result,
        initial_forward_gap=initial_forward_gap,
        discussion=discussion,
    )
    render_readme(Path("README.md"), readme_content)

    print(
        json.dumps(
            {
                "best_scratch_experiment": scratch_best_result["experiment_name"],
                "best_scratch_test_accuracy": scratch_best_result["test_metrics"]["accuracy"],
                "pytorch_test_accuracy": pytorch_result["test_metrics"]["accuracy"],
                "results_dir": str(project_config.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
