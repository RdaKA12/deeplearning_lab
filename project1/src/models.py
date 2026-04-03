from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.metrics_utils import compute_binary_classification_metrics


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_initial_parameters(layer_dims: list[int], seed: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    parameters: list[dict[str, np.ndarray]] = []
    for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:]):
        parameters.append(
            {
                "W": rng.normal(0.0, 0.01, size=(output_dim, input_dim)).astype(np.float64),
                "b": np.zeros((output_dim, 1), dtype=np.float64),
            }
        )
    return parameters


@dataclass
class TrainingSnapshot:
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float


class ScratchMLPClassifier:
    def __init__(
        self,
        layer_dims: list[int],
        learning_rate: float,
        max_epochs: int,
        patience: int,
        min_delta: float,
        threshold: float = 0.5,
        l2_lambda: float = 0.0,
        seed: int = 42,
        initial_parameters: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.threshold = threshold
        self.l2_lambda = l2_lambda
        self.seed = seed
        self.parameters = self._clone_parameters(
            initial_parameters if initial_parameters is not None else generate_initial_parameters(layer_dims, seed)
        )
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0
        self.epochs_trained = 0

    @staticmethod
    def _clone_parameters(parameters: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        return [{"W": layer["W"].copy(), "b": layer["b"].copy()} for layer in parameters]

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activation = X.T
        activations = [activation]
        for index, layer in enumerate(self.parameters):
            z_value = layer["W"] @ activation + layer["b"]
            activation = np.tanh(z_value) if index < len(self.parameters) - 1 else sigmoid(z_value)
            activations.append(activation)
        return activations[-1], activations

    def _compute_loss(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_target = y_true.T.astype(np.float64)
        clipped = np.clip(y_score, 1e-12, 1.0 - 1e-12)
        base_loss = -np.mean(y_target * np.log(clipped) + (1.0 - y_target) * np.log(1.0 - clipped))
        l2_term = 0.0
        if self.l2_lambda > 0:
            sample_count = y_true.shape[0]
            l2_term = (self.l2_lambda / (2.0 * sample_count)) * sum(np.sum(layer["W"] ** 2) for layer in self.parameters)
        return float(base_loss + l2_term)

    def _backward(self, X: np.ndarray, y_true: np.ndarray, activations: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        sample_count = X.shape[0]
        y_target = y_true.T.astype(np.float64)
        gradients: list[dict[str, np.ndarray]] = []
        delta = activations[-1] - y_target
        for layer_index in reversed(range(len(self.parameters))):
            current_layer = self.parameters[layer_index]
            previous_activation = activations[layer_index]
            dW = (delta @ previous_activation.T) / sample_count
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / sample_count) * current_layer["W"]
            db = np.sum(delta, axis=1, keepdims=True) / sample_count
            gradients.append({"W": dW, "b": db})
            if layer_index > 0:
                delta = (current_layer["W"].T @ delta) * (1.0 - activations[layer_index] ** 2)
        gradients.reverse()
        return gradients

    def _step(self, gradients: list[dict[str, np.ndarray]]) -> None:
        for layer, gradient in zip(self.parameters, gradients):
            layer["W"] -= self.learning_rate * gradient["W"]
            layer["b"] -= self.learning_rate * gradient["b"]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "ScratchMLPClassifier":
        best_parameters = self._clone_parameters(self.parameters)
        best_val_loss = float("inf")
        wait = 0
        for epoch in range(1, self.max_epochs + 1):
            _, activations = self._forward(X_train)
            self._step(self._backward(X_train, y_train, activations))

            train_scores, _ = self._forward(X_train)
            val_scores, _ = self._forward(X_val)
            train_predictions = (train_scores >= self.threshold).astype(np.int64).reshape(-1, 1)
            val_predictions = (val_scores >= self.threshold).astype(np.int64).reshape(-1, 1)
            snapshot = TrainingSnapshot(
                epoch=epoch,
                train_loss=self._compute_loss(y_train, train_scores),
                val_loss=self._compute_loss(y_val, val_scores),
                train_accuracy=compute_binary_classification_metrics(y_train, train_predictions)["accuracy"],
                val_accuracy=compute_binary_classification_metrics(y_val, val_predictions)["accuracy"],
            )
            self.history.append(snapshot.__dict__)

            if snapshot.val_loss < best_val_loss - self.min_delta:
                best_val_loss = snapshot.val_loss
                best_parameters = self._clone_parameters(self.parameters)
                self.best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    self.epochs_trained = epoch
                    self.parameters = best_parameters
                    return self

        self.epochs_trained = self.max_epochs
        self.parameters = best_parameters
        if self.best_epoch == 0:
            self.best_epoch = self.max_epochs
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores, _ = self._forward(X)
        return scores.T.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(np.int64)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        predictions = self.predict(X)
        scores = self.predict_proba(X)
        metrics = compute_binary_classification_metrics(y_true, predictions)
        metrics["loss"] = self._compute_loss(y_true, scores.T)
        return metrics


class TorchMLPNetwork(nn.Module):
    def __init__(self, layer_dims: list[int]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=True) for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:])]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for index, layer in enumerate(self.layers):
            output = layer(output)
            output = torch.tanh(output) if index < len(self.layers) - 1 else torch.sigmoid(output)
        return output


class TorchMLPClassifier:
    def __init__(
        self,
        layer_dims: list[int],
        learning_rate: float,
        max_epochs: int,
        patience: int,
        min_delta: float,
        threshold: float = 0.5,
        l2_lambda: float = 0.0,
        seed: int = 42,
        initial_parameters: list[dict[str, np.ndarray]] | None = None,
    ) -> None:
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.threshold = threshold
        self.l2_lambda = l2_lambda
        self.seed = seed
        torch.manual_seed(seed)
        self.model = TorchMLPNetwork(layer_dims).double()
        if initial_parameters is not None:
            self._load_initial_parameters(initial_parameters)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0
        self.epochs_trained = 0

    def _load_initial_parameters(self, initial_parameters: list[dict[str, np.ndarray]]) -> None:
        with torch.no_grad():
            for layer, initial_layer in zip(self.model.layers, initial_parameters):
                layer.weight.copy_(torch.from_numpy(initial_layer["W"]))
                layer.bias.copy_(torch.from_numpy(initial_layer["b"].reshape(-1)))

    def _loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(predictions, min=1e-12, max=1.0 - 1e-12)
        base_loss = -(targets * torch.log(clipped) + (1.0 - targets) * torch.log(1.0 - clipped)).mean()
        if self.l2_lambda <= 0:
            return base_loss
        sample_count = targets.shape[0]
        l2_term = 0.0
        for layer in self.model.layers:
            l2_term = l2_term + torch.sum(layer.weight.pow(2))
        return base_loss + (self.l2_lambda / (2.0 * sample_count)) * l2_term

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> "TorchMLPClassifier":
        X_train_tensor = torch.from_numpy(X_train).double()
        y_train_tensor = torch.from_numpy(y_train.astype(np.float64)).double()
        X_val_tensor = torch.from_numpy(X_val).double()
        y_val_tensor = torch.from_numpy(y_val.astype(np.float64)).double()

        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        wait = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            train_predictions = self.model(X_train_tensor)
            self._loss(train_predictions, y_train_tensor).backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                train_predictions = self.model(X_train_tensor)
                val_predictions = self.model(X_val_tensor)
                train_binary = (train_predictions >= self.threshold).to(torch.int64).cpu().numpy()
                val_binary = (val_predictions >= self.threshold).to(torch.int64).cpu().numpy()
                snapshot = TrainingSnapshot(
                    epoch=epoch,
                    train_loss=float(self._loss(train_predictions, y_train_tensor).item()),
                    val_loss=float(self._loss(val_predictions, y_val_tensor).item()),
                    train_accuracy=compute_binary_classification_metrics(y_train, train_binary)["accuracy"],
                    val_accuracy=compute_binary_classification_metrics(y_val, val_binary)["accuracy"],
                )
                self.history.append(snapshot.__dict__)

            if snapshot.val_loss < best_val_loss - self.min_delta:
                best_val_loss = snapshot.val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    self.epochs_trained = epoch
                    self.model.load_state_dict(best_state)
                    return self

        self.epochs_trained = self.max_epochs
        self.model.load_state_dict(best_state)
        if self.best_epoch == 0:
            self.best_epoch = self.max_epochs
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X).double()).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(np.int64)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        predictions = self.predict(X)
        y_tensor = torch.from_numpy(y_true.astype(np.float64)).double()
        score_tensor = torch.from_numpy(self.predict_proba(X)).double()
        metrics = compute_binary_classification_metrics(y_true, predictions)
        metrics["loss"] = float(self._loss(score_tensor, y_tensor).item())
        return metrics

    def compare_initial_forward(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X).double()).cpu().numpy()
