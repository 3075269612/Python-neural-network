from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(output: np.ndarray) -> np.ndarray:
    return output * (1.0 - output)


class TinyMLP:
    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.8, size=(2, 4)).astype(np.float32)
        self.b1 = np.zeros((1, 4), dtype=np.float32)
        self.w2 = rng.normal(0.0, 0.8, size=(4, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = sigmoid(x @ self.w1 + self.b1)
        y_hat = sigmoid(h @ self.w2 + self.b2)
        return h, y_hat

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> list[float]:
        losses: list[float] = []
        eps = 1e-8

        for _ in range(epochs):
            h, y_hat = self.forward(x)
            loss = -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)).mean()
            losses.append(float(loss))

            grad_logits = (y_hat - y) / len(x)
            grad_w2 = h.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0, keepdims=True)

            grad_h = grad_logits @ self.w2.T
            grad_z1 = grad_h * sigmoid_grad(h)
            grad_w1 = x.T @ grad_z1
            grad_b1 = grad_z1.sum(axis=0, keepdims=True)

            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1

        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        _, y_hat = self.forward(x)
        return (y_hat >= 0.5).astype(np.int64)


def save_loss_plot(losses: list[float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_title("XOR MLP Training Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("binary cross entropy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny MLP on XOR.")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    model = TinyMLP(seed=args.seed)
    losses = model.train(x, y, epochs=args.epochs, lr=args.lr)
    preds = model.predict(x)

    project_root = Path(__file__).resolve().parents[2]
    figure_path = project_root / "outputs" / "figures" / "ch1_xor_loss.png"
    save_loss_plot(losses, figure_path)

    print(f"final_loss={losses[-1]:.6f}")
    print("xor_predictions=")
    for features, pred, target in zip(x, preds.ravel(), y.ravel()):
        print(f"  x={features.tolist()} pred={int(pred)} target={int(target)}")
    print(f"saved_figure={figure_path}")


if __name__ == "__main__":
    main()
