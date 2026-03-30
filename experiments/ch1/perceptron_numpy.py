from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, n_features: int) -> None:
        self.weights = np.zeros(n_features, dtype=np.float32)
        self.bias = 0.0

    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_raw(x) >= 0).astype(np.int64)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> list[int]:
        mistakes_history: list[int] = []
        for _ in range(epochs):
            mistakes = 0
            for features, target in zip(x, y):
                pred = 1 if np.dot(features, self.weights) + self.bias >= 0 else 0
                update = lr * (target - pred)
                if update != 0:
                    mistakes += 1
                self.weights += update * features
                self.bias += update
            mistakes_history.append(mistakes)
            if mistakes == 0:
                break
        return mistakes_history


def make_toy_data(samples_per_class: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_0 = rng.normal(loc=(-2.0, -2.0), scale=0.8, size=(samples_per_class, 2))
    class_1 = rng.normal(loc=(2.0, 2.0), scale=0.8, size=(samples_per_class, 2))
    x = np.vstack([class_0, class_1]).astype(np.float32)
    y = np.concatenate(
        [np.zeros(samples_per_class, dtype=np.int64), np.ones(samples_per_class, dtype=np.int64)]
    )

    indices = rng.permutation(len(x))
    return x[indices], y[indices]


def save_decision_boundary_plot(
    x: np.ndarray,
    y: np.ndarray,
    model: Perceptron,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[y == 0, 0], x[y == 0, 1], label="class 0", alpha=0.75)
    ax.scatter(x[y == 1, 0], x[y == 1, 1], label="class 1", alpha=0.75)

    x_min, x_max = x[:, 0].min() - 1.0, x[:, 0].max() + 1.0
    line_x = np.linspace(x_min, x_max, 200)

    if abs(model.weights[1]) > 1e-8:
        line_y = -(model.weights[0] * line_x + model.bias) / model.weights[1]
        ax.plot(line_x, line_y, color="black", linewidth=2, label="decision boundary")

    ax.set_title("Perceptron on Toy Dataset")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a perceptron on a toy dataset.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    figure_path = project_root / "outputs" / "figures" / "ch1_perceptron_boundary.png"

    x, y = make_toy_data(seed=args.seed)
    model = Perceptron(n_features=x.shape[1])
    history = model.fit(x, y, epochs=args.epochs, lr=args.lr)

    preds = model.predict(x)
    acc = (preds == y).mean()

    print(f"epochs_run={len(history)}")
    print(f"final_mistakes={history[-1]}")
    print(f"train_accuracy={acc:.4f}")
    print(f"weights={model.weights}")
    print(f"bias={model.bias:.4f}")

    save_decision_boundary_plot(x, y, model, figure_path)
    print(f"saved_figure={figure_path}")


if __name__ == "__main__":
    main()
