from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, n_features: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.05, size=(n_features,)).astype(np.float32)
        self.bias = np.float32(0.0)

    def score(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) >= 0.0).astype(np.int64)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> list[int]:
        mistakes_history: list[int] = []

        for _ in range(epochs):
            mistakes = 0
            for features, target in zip(x, y):
                pred = 1 if np.dot(features, self.weights) + self.bias >= 0.0 else 0

                # Perceptron rule: only update when prediction is wrong.
                update = lr * (target - pred)
                if update != 0.0:
                    mistakes += 1
                    self.weights += update * features
                    self.bias += update

            mistakes_history.append(mistakes)
            if mistakes == 0:
                break

        return mistakes_history


def make_linearly_separable_data(
    samples_per_class: int = 120,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    class_0 = rng.normal(loc=(-2.0, -1.5), scale=0.75, size=(samples_per_class, 2))
    class_1 = rng.normal(loc=(2.0, 1.5), scale=0.75, size=(samples_per_class, 2))

    x = np.vstack([class_0, class_1]).astype(np.float32)
    y = np.concatenate(
        [
            np.zeros(samples_per_class, dtype=np.int64),
            np.ones(samples_per_class, dtype=np.int64),
        ]
    )

    indices = rng.permutation(len(x))
    return x[indices], y[indices]


def split_train_test(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_size = int(len(x) * test_ratio)
    if test_size <= 0 or test_size >= len(x):
        raise ValueError("test_ratio must produce a non-empty train set and test set.")

    x_test, y_test = x[:test_size], y[:test_size]
    x_train, y_train = x[test_size:], y[test_size:]
    return x_train, y_train, x_test, y_test


def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float((preds == targets).mean())


def save_experiment_figure(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model: Perceptron,
    mistakes_history: list[int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    ax0.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], label="class 0", alpha=0.75)
    ax0.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], label="class 1", alpha=0.75)

    x_min, x_max = x_train[:, 0].min() - 1.0, x_train[:, 0].max() + 1.0
    line_x = np.linspace(x_min, x_max, 300)

    if abs(float(model.weights[1])) > 1e-8:
        line_y = -(model.weights[0] * line_x + model.bias) / model.weights[1]
        ax0.plot(line_x, line_y, color="black", linewidth=2, label="decision boundary")
    elif abs(float(model.weights[0])) > 1e-8:
        x_vertical = -float(model.bias) / float(model.weights[0])
        ax0.axvline(x_vertical, color="black", linewidth=2, label="decision boundary")

    ax0.set_title("Experiment 1.1: Perceptron Decision Boundary")
    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(range(1, len(mistakes_history) + 1), mistakes_history, marker="o")
    ax1.set_title("Training Mistakes per Epoch")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mistakes")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_sample_predictions(
    model: Perceptron,
    x: np.ndarray,
    y: np.ndarray,
    sample_count: int = 8,
) -> None:
    count = min(sample_count, len(x))
    print("sample_predictions=")
    for idx in range(count):
        score = float(model.score(x[idx : idx + 1])[0])
        pred = int(model.predict(x[idx : idx + 1])[0])
        target = int(y[idx])
        x1, x2 = x[idx]
        print(
            f"  id={idx:02d} x=({x1:.3f}, {x2:.3f}) "
            f"score={score:.3f} pred={pred} target={target}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1.1: Perceptron on linearly separable data.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    figure_path = project_root / "outputs" / "figures" / "ch1_exp1_1_perceptron.png"

    x, y = make_linearly_separable_data(samples_per_class=args.samples_per_class, seed=args.seed)
    x_train, y_train, x_test, y_test = split_train_test(x, y, test_ratio=args.test_ratio)

    model = Perceptron(n_features=x_train.shape[1], seed=args.seed)
    mistakes_history = model.fit(x_train, y_train, epochs=args.epochs, lr=args.lr)

    train_acc = accuracy(model.predict(x_train), y_train)
    test_acc = accuracy(model.predict(x_test), y_test)

    print("=== Experiment 1.1: Perceptron Linear Classifier ===")
    print(f"epochs_run={len(mistakes_history)}")
    print(f"final_epoch_mistakes={mistakes_history[-1]}")
    print(f"train_accuracy={train_acc:.4f}")
    print(f"test_accuracy={test_acc:.4f}")
    print(f"weights={model.weights}")
    print(f"bias={float(model.bias):.4f}")

    print_sample_predictions(model, x_test, y_test)
    save_experiment_figure(x_train, y_train, model, mistakes_history, figure_path)
    print(f"saved_figure={figure_path}")


if __name__ == "__main__":
    main()