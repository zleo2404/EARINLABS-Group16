"""Lab 5, Group A: MLP activation comparison on MNIST.

Authors:
    Anatoli Abahumna
    Leonardo Meloni
"""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_DIR = PROJECT_DIR / "data"

SEEDS = [7, 21, 42]
BATCH_SIZE = 128
EPOCHS = 18
LEARNING_RATE = 0.5

BASE_ARCHITECTURE = [784, 64, 32, 10]
SECOND_ARCHITECTURE = [784, 128, 64, 10]


class Layer(ABC):
    """Basic building block of the neural network."""

    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through the layer."""

    @abstractmethod
    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        """Backward propagation through the layer."""

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    """Dense layer: y = xW + b."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rng = rng if rng is not None else np.random.default_rng()

        limit = np.sqrt(6.0 / (input_size + output_size))
        self.weights = self.rng.uniform(
            -limit,
            limit,
            size=(input_size, output_size),
        ).astype(np.float64)
        self.bias = np.zeros((1, output_size), dtype=np.float64)
        self.input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")

        weights_grad = self.input.T @ output_error_derivative
        bias_grad = np.sum(output_error_derivative, axis=0, keepdims=True)
        input_error = output_error_derivative @ self.weights.T

        self.weights -= self.learning_rate * weights_grad
        self.bias -= self.learning_rate * bias_grad
        return input_error


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError("forward must be called before backward")
        return output_error_derivative * (1.0 - self.output**2)


class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -500.0, 500.0)
        self.output = 1.0 / (1.0 + np.exp(-clipped))
        return self.output

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        if self.output is None:
            raise RuntimeError("forward must be called before backward")
        return output_error_derivative * self.output * (1.0 - self.output)


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0.0, x)

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        return output_error_derivative * (self.input > 0.0)


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha
        self.input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.where(x > 0.0, x, self.alpha * x)

    def backward(self, output_error_derivative: np.ndarray) -> np.ndarray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")
        slope = np.where(self.input > 0.0, 1.0, self.alpha)
        return output_error_derivative * slope


class Loss:
    def __init__(
        self,
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        loss_function_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_function(y_pred, y_true)

    def loss_derivative(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        return self.loss_function_derivative(y_pred, y_true)


class Network:
    def __init__(self, layers: list[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss: Loss | None = None

    def compile(self, loss: Loss) -> None:
        self.loss = loss
        for layer in self.layers:
            layer.learning_rate = self.learning_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        learning_rate: float,
        batch_size: int = 128,
        seed: int = 0,
        verbose: int = 0,
    ) -> list[float]:
        if self.loss is None:
            raise RuntimeError("compile must be called before fit")

        self.learning_rate = learning_rate
        for layer in self.layers:
            layer.learning_rate = learning_rate

        rng = np.random.default_rng(seed)
        losses = []
        n_samples = x_train.shape[0]

        for epoch in range(epochs):
            indices = rng.permutation(n_samples)
            epoch_losses = []

            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start:start + batch_size]
                x_batch = x_train[batch_indices]
                y_batch = y_train[batch_indices]

                y_pred = self(x_batch)
                epoch_losses.append(self.loss.loss(y_pred, y_batch))

                error = self.loss.loss_derivative(y_pred, y_batch)
                for layer in reversed(self.layers):
                    error = layer.backward(error)

            mean_loss = float(np.mean(epoch_losses))
            losses.append(mean_loss)

            if verbose:
                print(f"epoch {epoch + 1:02d}/{epochs}: loss={mean_loss:.5f}")

        return losses


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2.0 * (y_pred - y_true) / y_pred.shape[0]


def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float64)[labels]


def build_network(
    architecture: list[int],
    hidden_activation: type[Layer],
    seed: int,
    learning_rate: float,
) -> Network:
    rng = np.random.default_rng(seed)
    layers: list[Layer] = []

    for i in range(len(architecture) - 1):
        layers.append(FullyConnected(architecture[i], architecture[i + 1], rng))
        if i < len(architecture) - 2:
            layers.append(hidden_activation())
        else:
            layers.append(Sigmoid())

    network = Network(layers, learning_rate)
    network.compile(Loss(mse, mse_derivative))
    return network


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        data_home=str(DATA_DIR),
    )
    x = dataset.data.astype(np.float64) / 255.0
    labels = dataset.target.astype(int)

    x_train_full, x_test_full, labels_train_full, labels_test_full = (
        train_test_split(
            x,
            labels,
            test_size=0.2,
            random_state=0,
            stratify=labels,
        )
    )

    return (
        x_train_full,
        one_hot(labels_train_full),
        x_test_full,
        one_hot(labels_test_full),
    )


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    predicted = np.argmax(y_pred, axis=1)
    expected = np.argmax(y_true, axis=1)
    return float(np.mean(predicted == expected))


def evaluate(network: Network, x_test: np.ndarray, y_test: np.ndarray) -> float:
    return accuracy(network(x_test), y_test)


def mean_curve(curves: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(curves, dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(
    rows: list[dict[str, object]],
    group_key: str,
) -> list[dict[str, object]]:
    groups = sorted({str(row[group_key]) for row in rows})
    summary = []

    for group in groups:
        accuracies = np.array(
            [
                float(row["test_accuracy"])
                for row in rows
                if str(row[group_key]) == group
            ]
        )
        summary.append(
            {
                group_key: group,
                "mean_accuracy": accuracies.mean(),
                "std_accuracy": accuracies.std(ddof=1),
            }
        )

    return summary


def plot_activation_losses(
    loss_curves: dict[str, list[list[float]]],
    path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, EPOCHS + 1)

    for activation_name, curves in loss_curves.items():
        mean_loss, std_loss = mean_curve(curves)
        plt.plot(epochs, mean_loss, label=activation_name)
        plt.fill_between(
            epochs,
            mean_loss - std_loss,
            mean_loss + std_loss,
            alpha=0.12,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("MNIST training loss by hidden activation")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_architecture_losses(
    loss_curves: dict[str, list[list[float]]],
    path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, EPOCHS + 1)

    for architecture_name, curves in loss_curves.items():
        mean_loss, std_loss = mean_curve(curves)
        plt.plot(epochs, mean_loss, label=architecture_name)
        plt.fill_between(
            epochs,
            mean_loss - std_loss,
            mean_loss + std_loss,
            alpha=0.12,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training MSE")
    plt.title("Architecture comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_activation_comparison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, list[list[float]]]]:
    activations: dict[str, type[Layer]] = {
        "Tanh": Tanh,
        "Sigmoid": Sigmoid,
        "ReLU": ReLU,
        "LeakyReLU": LeakyReLU,
    }
    rows = []
    loss_curves: dict[str, list[list[float]]] = {
        name: [] for name in activations
    }

    for activation_name, activation_class in activations.items():
        for seed in SEEDS:
            print(f"activation={activation_name}, seed={seed}")
            network = build_network(
                BASE_ARCHITECTURE,
                activation_class,
                seed,
                LEARNING_RATE,
            )
            losses = network.fit(
                x_train,
                y_train,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                seed=seed,
            )
            test_accuracy = evaluate(network, x_test, y_test)
            rows.append(
                {
                    "activation": activation_name,
                    "seed": seed,
                    "test_accuracy": test_accuracy,
                    "final_train_loss": losses[-1],
                }
            )
            loss_curves[activation_name].append(losses)
            print(
                f"  accuracy={test_accuracy:.4f}, "
                f"final_loss={losses[-1]:.4f}"
            )

    return rows, loss_curves


def run_architecture_comparison(
    best_activation_name: str,
    best_activation_class: type[Layer],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, list[list[float]]]]:
    architectures = {
        "64-32": BASE_ARCHITECTURE,
        "128-64": SECOND_ARCHITECTURE,
    }
    rows = []
    loss_curves: dict[str, list[list[float]]] = {
        name: [] for name in architectures
    }

    for architecture_name, architecture in architectures.items():
        for seed in SEEDS:
            print(
                f"architecture={architecture_name}, "
                f"activation={best_activation_name}, seed={seed}"
            )
            network = build_network(
                architecture,
                best_activation_class,
                seed,
                LEARNING_RATE,
            )
            losses = network.fit(
                x_train,
                y_train,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                seed=seed,
            )
            test_accuracy = evaluate(network, x_test, y_test)
            rows.append(
                {
                    "architecture": architecture_name,
                    "activation": best_activation_name,
                    "seed": seed,
                    "test_accuracy": test_accuracy,
                    "final_train_loss": losses[-1],
                }
            )
            loss_curves[architecture_name].append(losses)
            print(
                f"  accuracy={test_accuracy:.4f}, "
                f"final_loss={losses[-1]:.4f}"
            )

    return rows, loss_curves


def run_implementation_tests() -> None:
    x = np.random.default_rng(123).normal(size=(5, 4))
    y = one_hot(np.array([0, 1, 2, 1, 0]), n_classes=3)
    network = build_network([4, 6, 5, 3], ReLU, seed=123, learning_rate=0.1)
    prediction = network(x)

    if prediction.shape != (5, 3):
        raise AssertionError("batched forward pass returned the wrong shape")

    losses = network.fit(
        x,
        y,
        epochs=2,
        learning_rate=0.1,
        batch_size=2,
        seed=123,
    )
    if len(losses) != 2:
        raise AssertionError("fit did not return one loss per epoch")

    if not np.isfinite(losses).all():
        raise AssertionError("training produced a non-finite loss")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_implementation_tests()
    print("implementation tests passed")

    print("loading MNIST")
    x_train, y_train, x_test, y_test = load_mnist()
    print(
        f"train={x_train.shape[0]} samples, test={x_test.shape[0]} samples, "
        f"features={x_train.shape[1]}"
    )

    activation_rows, activation_loss_curves = run_activation_comparison(
        x_train,
        y_train,
        x_test,
        y_test,
    )
    activation_summary = summarize_results(activation_rows, "activation")

    write_rows(OUTPUT_DIR / "activation_results.csv", activation_rows)
    write_rows(OUTPUT_DIR / "activation_summary.csv", activation_summary)
    plot_activation_losses(
        activation_loss_curves,
        OUTPUT_DIR / "activation_training_loss.png",
    )

    best_activation_row = max(
        activation_summary,
        key=lambda row: float(row["mean_accuracy"]),
    )
    best_activation_name = str(best_activation_row["activation"])
    activation_classes: dict[str, type[Layer]] = {
        "Tanh": Tanh,
        "Sigmoid": Sigmoid,
        "ReLU": ReLU,
        "LeakyReLU": LeakyReLU,
    }
    best_activation_class = activation_classes[best_activation_name]

    architecture_rows, architecture_loss_curves = run_architecture_comparison(
        best_activation_name,
        best_activation_class,
        x_train,
        y_train,
        x_test,
        y_test,
    )
    architecture_summary = summarize_results(
        architecture_rows,
        "architecture",
    )

    write_rows(OUTPUT_DIR / "architecture_results.csv", architecture_rows)
    write_rows(OUTPUT_DIR / "architecture_summary.csv", architecture_summary)
    plot_architecture_losses(
        architecture_loss_curves,
        OUTPUT_DIR / "architecture_training_loss.png",
    )

    print("\nactivation summary")
    for row in activation_summary:
        print(
            f"{row['activation']}: "
            f"{float(row['mean_accuracy']):.4f} "
            f"+/- {float(row['std_accuracy']):.4f}"
        )

    print("\narchitecture summary")
    for row in architecture_summary:
        print(
            f"{row['architecture']}: "
            f"{float(row['mean_accuracy']):.4f} "
            f"+/- {float(row['std_accuracy']):.4f}"
        )


if __name__ == "__main__":
    main()
