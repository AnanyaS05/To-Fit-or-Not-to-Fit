from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .metrics import accuracy_score, macro_f1_score


@dataclass(slots=True)
class ManualMLPConfig:
    hidden_dims: tuple[int, ...] = (128, 64)
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 25
    patience: int = 5
    dropout: float = 0.10
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    seed: int = 42
    verbose: bool = True

    def __post_init__(self) -> None:
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")


class ManualMLPClassifier:
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        config: ManualMLPConfig | None = None,
        class_weights: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config or ManualMLPConfig()
        self.rng = np.random.default_rng(self.config.seed)

        if class_names is not None and len(class_names) != num_classes:
            raise ValueError("class_names length must match num_classes")
        self.class_names = class_names or [str(index) for index in range(num_classes)]
        self.class_weights = class_weights
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.weight_moments: list[np.ndarray] = []
        self.weight_velocities: list[np.ndarray] = []
        self.bias_moments: list[np.ndarray] = []
        self.bias_velocities: list[np.ndarray] = []
        self.history_: dict[str, list[float] | int] = {}
        self.best_epoch_: int = 0
        self.optimizer_step_: int = 0

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        layer_dims = [self.input_dim, *self.config.hidden_dims, self.num_classes]

        self.weights = []
        self.biases = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            weight = self.rng.normal(
                loc=0.0,
                scale=np.sqrt(2.0 / fan_in),
                size=(fan_in, fan_out),
            ).astype(np.float32)
            bias = np.zeros((1, fan_out), dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)

        self.weight_moments = [np.zeros_like(weight) for weight in self.weights]
        self.weight_velocities = [np.zeros_like(weight) for weight in self.weights]
        self.bias_moments = [np.zeros_like(bias) for bias in self.biases]
        self.bias_velocities = [np.zeros_like(bias) for bias in self.biases]
        self.optimizer_step_ = 0

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted).astype(np.float32, copy=False)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0).astype(np.float32, copy=False)

    def _forward(self, X: np.ndarray, training: bool) -> tuple[np.ndarray, dict[str, list[np.ndarray | None]]]:
        activations: list[np.ndarray] = [X.astype(np.float32, copy=False)]
        pre_activations: list[np.ndarray] = []
        dropout_masks: list[np.ndarray | None] = []

        current = activations[0]
        keep_prob = 1.0 - self.config.dropout

        for layer_idx in range(len(self.weights) - 1):
            z_values = current @ self.weights[layer_idx] + self.biases[layer_idx]
            pre_activations.append(z_values)
            current = self._relu(z_values)

            if training and self.config.dropout > 0.0:
                mask = self.rng.binomial(1, keep_prob, size=current.shape).astype(np.float32)
                current = (current * mask) / keep_prob
            else:
                mask = None

            dropout_masks.append(mask)
            activations.append(current)

        logits = current @ self.weights[-1] + self.biases[-1]
        pre_activations.append(logits)
        activations.append(logits)

        cache = {
            "activations": activations,
            "pre_activations": pre_activations,
            "dropout_masks": dropout_masks,
        }
        return logits, cache

    def _loss_and_output_gradient(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        probabilities = self._softmax(logits)
        batch_indices = np.arange(len(y_true))

        if self.class_weights is None:
            sample_weights = np.ones(len(y_true), dtype=np.float32)
        else:
            sample_weights = self.class_weights[y_true].astype(np.float32, copy=False)

        normalizer = float(np.sum(sample_weights))
        clipped = np.clip(probabilities[batch_indices, y_true], 1e-8, 1.0)
        data_loss = -np.sum(sample_weights * np.log(clipped)) / normalizer

        regularization = 0.5 * self.config.weight_decay * sum(
            float(np.sum(weight * weight)) for weight in self.weights
        )
        loss = float(data_loss + regularization)

        output_gradient = probabilities
        output_gradient[batch_indices, y_true] -= 1.0
        output_gradient *= (sample_weights / normalizer)[:, None]
        return loss, output_gradient.astype(np.float32, copy=False)

    def _backward(
        self,
        cache: dict[str, list[np.ndarray | None]],
        output_gradient: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = cache["activations"]
        pre_activations = cache["pre_activations"]
        dropout_masks = cache["dropout_masks"]

        weight_grads: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        bias_grads: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]

        delta = output_gradient
        last_hidden = activations[-2]
        weight_grads[-1] = last_hidden.T @ delta + (self.config.weight_decay * self.weights[-1])
        bias_grads[-1] = np.sum(delta, axis=0, keepdims=True)
        upstream = delta @ self.weights[-1].T

        keep_prob = 1.0 - self.config.dropout

        for layer_idx in range(len(self.weights) - 2, -1, -1):
            mask = dropout_masks[layer_idx]
            if mask is not None:
                upstream = (upstream * mask) / keep_prob

            relu_grad = (pre_activations[layer_idx] > 0).astype(np.float32)
            delta = upstream * relu_grad

            prev_activation = activations[layer_idx]
            weight_grads[layer_idx] = (
                prev_activation.T @ delta + (self.config.weight_decay * self.weights[layer_idx])
            )
            bias_grads[layer_idx] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx > 0:
                upstream = delta @ self.weights[layer_idx].T

        return weight_grads, bias_grads

    def _apply_gradients(
        self,
        weight_grads: list[np.ndarray],
        bias_grads: list[np.ndarray],
    ) -> None:
        self.optimizer_step_ += 1
        step = self.optimizer_step_

        for idx in range(len(self.weights)):
            self.weight_moments[idx] = (
                self.config.beta1 * self.weight_moments[idx]
                + (1.0 - self.config.beta1) * weight_grads[idx]
            )
            self.weight_velocities[idx] = (
                self.config.beta2 * self.weight_velocities[idx]
                + (1.0 - self.config.beta2) * np.square(weight_grads[idx])
            )
            self.bias_moments[idx] = (
                self.config.beta1 * self.bias_moments[idx]
                + (1.0 - self.config.beta1) * bias_grads[idx]
            )
            self.bias_velocities[idx] = (
                self.config.beta2 * self.bias_velocities[idx]
                + (1.0 - self.config.beta2) * np.square(bias_grads[idx])
            )

            weight_m_hat = self.weight_moments[idx] / (1.0 - self.config.beta1**step)
            weight_v_hat = self.weight_velocities[idx] / (1.0 - self.config.beta2**step)
            bias_m_hat = self.bias_moments[idx] / (1.0 - self.config.beta1**step)
            bias_v_hat = self.bias_velocities[idx] / (1.0 - self.config.beta2**step)

            self.weights[idx] -= self.config.learning_rate * (
                weight_m_hat / (np.sqrt(weight_v_hat) + self.config.epsilon)
            )
            self.biases[idx] -= self.config.learning_rate * (
                bias_m_hat / (np.sqrt(bias_v_hat) + self.config.epsilon)
            )

    def _compute_class_weights(self, y_train: np.ndarray) -> np.ndarray:
        counts = np.bincount(y_train, minlength=self.num_classes).astype(np.float32)
        safe_counts = np.where(counts == 0, 1.0, counts)
        weights = counts.sum() / (self.num_classes * safe_counts)
        return (weights / np.mean(weights)).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, list[float] | int]:
        if self.class_weights is None:
            self.class_weights = self._compute_class_weights(y_train)

        history: dict[str, list[float] | int] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_f1": [],
            "best_epoch": 0,
        }

        best_score = -np.inf
        best_weights: list[np.ndarray] | None = None
        best_biases: list[np.ndarray] | None = None
        patience_counter = 0
        sample_count = X_train.shape[0]

        for epoch in range(1, self.config.max_epochs + 1):
            permutation = self.rng.permutation(sample_count)
            epoch_losses: list[float] = []

            for start_idx in range(0, sample_count, self.config.batch_size):
                batch_indices = permutation[start_idx : start_idx + self.config.batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                logits, cache = self._forward(X_batch, training=True)
                loss, output_gradient = self._loss_and_output_gradient(logits, y_batch)
                weight_grads, bias_grads = self._backward(cache, output_gradient)
                self._apply_gradients(weight_grads, bias_grads)
                epoch_losses.append(loss)

            train_loss = float(np.mean(epoch_losses))

            val_logits, _ = self._forward(X_val, training=False)
            val_loss, _ = self._loss_and_output_gradient(val_logits, y_val)
            val_predictions = np.argmax(val_logits, axis=1)
            val_accuracy = float(accuracy_score(y_val, val_predictions))
            val_macro_f1 = float(
                macro_f1_score(y_val, val_predictions, labels=np.arange(self.num_classes))
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(val_accuracy)
            history["val_macro_f1"].append(val_macro_f1)

            if self.config.verbose:
                print(
                    "Epoch "
                    f"{epoch:02d} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | val_accuracy={val_accuracy:.4f} | "
                    f"val_macro_f1={val_macro_f1:.4f}"
                )

            if val_macro_f1 > best_score:
                best_score = val_macro_f1
                best_weights = [weight.copy() for weight in self.weights]
                best_biases = [bias.copy() for bias in self.biases]
                self.best_epoch_ = epoch
                history["best_epoch"] = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    if self.config.verbose:
                        print("Early stopping triggered.")
                    break

        if best_weights is not None and best_biases is not None:
            self.weights = best_weights
            self.biases = best_biases

        self.history_ = history
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(X, training=False)
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        predictions = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(macro_f1_score(y_true, predictions, labels=np.arange(self.num_classes))),
        }

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "config": self.config_dict(),
            "best_epoch": self.best_epoch_,
            "optimizer_step": self.optimizer_step_,
            "history": self.history_,
            "has_class_weights": self.class_weights is not None,
            "num_layers": len(self.weights),
        }
        payload: dict[str, object] = {
            "metadata": np.asarray(json.dumps(metadata)),
        }
        for idx, weight in enumerate(self.weights):
            payload[f"weight_{idx}"] = weight
        for idx, bias in enumerate(self.biases):
            payload[f"bias_{idx}"] = bias
        if self.class_weights is not None:
            payload["class_weights"] = self.class_weights

        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ManualMLPClassifier":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            config = ManualMLPConfig(**metadata["config"])
            class_weights = data["class_weights"] if metadata["has_class_weights"] else None
            model = cls(
                input_dim=int(metadata["input_dim"]),
                num_classes=int(metadata["num_classes"]),
                config=config,
                class_weights=class_weights,
                class_names=metadata.get("class_names"),
            )
            model.weights = [
                data[f"weight_{idx}"].astype(np.float32, copy=True)
                for idx in range(int(metadata["num_layers"]))
            ]
            model.biases = [
                data[f"bias_{idx}"].astype(np.float32, copy=True)
                for idx in range(int(metadata["num_layers"]))
            ]
            model.weight_moments = [np.zeros_like(weight) for weight in model.weights]
            model.weight_velocities = [np.zeros_like(weight) for weight in model.weights]
            model.bias_moments = [np.zeros_like(bias) for bias in model.biases]
            model.bias_velocities = [np.zeros_like(bias) for bias in model.biases]
            model.best_epoch_ = int(metadata["best_epoch"])
            model.optimizer_step_ = int(metadata["optimizer_step"])
            model.history_ = metadata["history"]

        return model
