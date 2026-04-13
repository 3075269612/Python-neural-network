from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(output: np.ndarray) -> np.ndarray:
    return output * (1.0 - output)


class TinyMLP:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)

        w1_std = np.sqrt(1.0 / input_dim)
        w2_std = np.sqrt(1.0 / hidden_dim)

        self.w1 = rng.normal(0.0, w1_std, size=(input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.w2 = rng.normal(0.0, w2_std, size=(hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros((1, 1), dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        h = sigmoid(x @ self.w1 + self.b1)
        y_hat = sigmoid(h @ self.w2 + self.b2)
        cache = {"x": x, "h": h, "y_hat": y_hat}
        return y_hat, cache

    @staticmethod
    def compute_bce_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-8
        loss = -(y * np.log(y_hat + eps) + (1.0 - y) * np.log(1.0 - y_hat + eps)).mean()
        return float(loss)

    def backward(self, y: np.ndarray, cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = cache["x"]
        h = cache["h"]
        y_hat = cache["y_hat"]

        batch_size = x.shape[0]
        grad_logits = (y_hat - y) / batch_size

        grad_w2 = h.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0, keepdims=True)

        grad_h = grad_logits @ self.w2.T
        grad_z1 = grad_h * sigmoid_grad(h)
        grad_w1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0, keepdims=True)

        return {
            "w2": grad_w2,
            "b2": grad_b2,
            "w1": grad_w1,
            "b1": grad_b1,
        }

    def update_weights(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.w2 -= lr * grads["w2"]
        self.b2 -= lr * grads["b2"]
        self.w1 -= lr * grads["w1"]
        self.b1 -= lr * grads["b1"]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        y_hat, _ = self.forward(x)
        return y_hat

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_hat = self.predict_proba(x)
        return (y_hat >= 0.5).astype(np.int64)


def make_noisy_xor_data(
    samples_per_quadrant: int,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )

    labels = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    samples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for center, label in zip(centers, labels):
        cluster = rng.normal(loc=center, scale=noise, size=(samples_per_quadrant, 2)).astype(np.float32)
        cluster_labels = np.full((samples_per_quadrant, 1), label.item(), dtype=np.float32)
        samples.append(cluster)
        targets.append(cluster_labels)

    x = np.vstack(samples).astype(np.float32)
    y = np.vstack(targets).astype(np.float32)

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


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def compute_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    return float((preds.ravel() == targets.ravel()).mean())


def train_model(
    model: TinyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    log_every: int,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_proba, cache = model.forward(x_train)
        train_loss = model.compute_bce_loss(train_proba, y_train)

        grads = model.backward(y_train, cache)
        model.update_weights(grads, lr=lr)

        train_acc = compute_accuracy(model.predict(x_train), y_train)
        test_proba = model.predict_proba(x_test)
        test_loss = model.compute_bce_loss(test_proba, y_test)
        test_acc = compute_accuracy(model.predict(x_test), y_test)

        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(metrics)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    return history


def save_training_log(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for row in history:
            writer.writerow(
                [
                    int(row["epoch"]),
                    row["train_loss"],
                    row["train_acc"],
                    row["test_loss"],
                    row["test_acc"],
                ]
            )


def save_experiment_figure(
    history: list[dict[str, float]],
    model: TinyMLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(item["epoch"]) for item in history]
    train_losses = [item["train_loss"] for item in history]
    test_losses = [item["test_loss"] for item in history]
    train_accs = [item["train_acc"] for item in history]
    test_accs = [item["test_acc"] for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax0 = axes[0]
    ax0.plot(epochs, train_losses, label="train loss")
    ax0.plot(epochs, test_losses, label="test loss")
    ax0.set_title("Experiment 1.2: BCE Loss")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(epochs, train_accs, label="train acc")
    ax1.plot(epochs, test_accs, label="test acc")
    ax1.set_title("Accuracy Curve")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend()

    ax2 = axes[2]
    x_min, x_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
    y_min, y_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200, dtype=np.float32),
        np.linspace(y_min, y_max, 200, dtype=np.float32),
    )

    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    zz = model.predict_proba(grid).reshape(xx.shape)
    contour = ax2.contourf(xx, yy, zz, levels=30, cmap="RdBu", alpha=0.65)
    fig.colorbar(contour, ax=ax2, fraction=0.046, pad=0.04)
    ax2.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.5)

    mask = y_train.ravel() > 0.5
    ax2.scatter(x_train[~mask, 0], x_train[~mask, 1], label="class 0", edgecolor="k", s=20)
    ax2.scatter(x_train[mask, 0], x_train[mask, 1], label="class 1", edgecolor="k", s=20)
    ax2.set_title("Decision Boundary (Train)")
    ax2.set_xlabel("x1 (standardized)")
    ax2.set_ylabel("x2 (standardized)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_sample_predictions(
    model: TinyMLP,
    x_test: np.ndarray,
    y_test: np.ndarray,
    sample_count: int = 10,
) -> None:
    count = min(sample_count, len(x_test))
    probs = model.predict_proba(x_test[:count]).ravel()
    preds = (probs >= 0.5).astype(np.int64)

    print("sample_predictions=")
    for idx in range(count):
        x1, x2 = x_test[idx]
        print(
            f"  id={idx:02d} x=({x1:.3f}, {x2:.3f}) "
            f"proba={probs[idx]:.3f} pred={int(preds[idx])} target={int(y_test[idx, 0])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 1.2: three-layer network and backpropagation on noisy XOR."
    )
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--samples-per-quadrant", type=int, default=120)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    x, y = make_noisy_xor_data(
        samples_per_quadrant=args.samples_per_quadrant,
        noise=args.noise,
        seed=args.seed,
    )
    x_train, y_train, x_test, y_test = split_train_test(x, y, test_ratio=args.test_ratio)
    x_train, x_test = standardize_train_test(x_train, x_test)

    model = TinyMLP(input_dim=2, hidden_dim=args.hidden_dim, seed=args.seed)
    history = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=args.epochs,
        lr=args.lr,
        log_every=max(args.log_every, 1),
    )

    train_acc = compute_accuracy(model.predict(x_train), y_train)
    test_acc = compute_accuracy(model.predict(x_test), y_test)
    final_train_loss = history[-1]["train_loss"]
    final_test_loss = history[-1]["test_loss"]

    project_root = Path(__file__).resolve().parents[2]
    figure_path = project_root / "outputs" / "figures" / "ch1_exp1_2_mlp_backprop.png"
    log_path = project_root / "outputs" / "logs" / "ch1_exp1_2_metrics.csv"

    save_experiment_figure(history, model, x_train, y_train, figure_path)
    save_training_log(history, log_path)

    print("=== Experiment 1.2: Three-Layer Neural Network with Backpropagation ===")
    print(f"train_size={len(x_train)} test_size={len(x_test)}")
    print(f"final_train_loss={final_train_loss:.6f}")
    print(f"final_test_loss={final_test_loss:.6f}")
    print(f"final_train_accuracy={train_acc:.4f}")
    print(f"final_test_accuracy={test_acc:.4f}")
    print("w1=")
    print(np.array2string(model.w1, precision=4))
    print("b1=")
    print(np.array2string(model.b1, precision=4))
    print("w2=")
    print(np.array2string(model.w2, precision=4))
    print("b2=")
    print(np.array2string(model.b2, precision=4))

    print_sample_predictions(model, x_test, y_test, sample_count=10)
    print(f"saved_figure={figure_path}")
    print(f"saved_log={log_path}")


if __name__ == "__main__":
    main()
