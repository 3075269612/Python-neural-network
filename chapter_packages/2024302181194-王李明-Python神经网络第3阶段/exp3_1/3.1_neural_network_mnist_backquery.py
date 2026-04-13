from __future__ import annotations

import argparse
import csv
import struct
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.special


def load_mnist_idx(images_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with labels_path.open("rb") as lb_file:
        _, n_labels = struct.unpack(">II", lb_file.read(8))
        labels = np.frombuffer(lb_file.read(), dtype=np.uint8)

    with images_path.open("rb") as img_file:
        _, n_images, rows, cols = struct.unpack(">IIII", img_file.read(16))
        images = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(n_images, rows * cols)

    if n_images != n_labels:
        raise ValueError("MNIST image/label count mismatch.")

    return images, labels


def load_mnist_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    labels = np.asarray([int(line.split(",")[0]) for line in lines], dtype=np.uint8)
    images = np.asarray([line.split(",")[1:] for line in lines], dtype=np.uint8)
    return images, labels


def build_dataset(
    project_root: Path,
    dataset: str,
    train_limit: int,
    test_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "idx":
        raw_dir = project_root / "data" / "raw" / "MNIST" / "raw"
        train_images, train_labels = load_mnist_idx(
            raw_dir / "train-images-idx3-ubyte",
            raw_dir / "train-labels-idx1-ubyte",
        )
        test_images, test_labels = load_mnist_idx(
            raw_dir / "t10k-images-idx3-ubyte",
            raw_dir / "t10k-labels-idx1-ubyte",
        )
    else:
        csv_dir = project_root / "data" / "raw" / "book_mnist_csv"
        train_images, train_labels = load_mnist_csv(csv_dir / "mnist_train.csv")
        test_images, test_labels = load_mnist_csv(csv_dir / "mnist_test.csv")

    if train_limit > 0:
        train_images = train_images[:train_limit]
        train_labels = train_labels[:train_limit]
    if test_limit > 0:
        test_images = test_images[:test_limit]
        test_labels = test_labels[:test_limit]

    return train_images, train_labels, test_images, test_labels


def scale_inputs(pixel_values: list[str] | np.ndarray) -> np.ndarray:
    values = np.asarray(pixel_values, dtype=np.float32)
    return (values / 255.0 * 0.99) + 0.01


def one_hot_target(label: int, output_nodes: int) -> np.ndarray:
    targets = np.zeros(output_nodes, dtype=np.float32) + 0.01
    targets[int(label)] = 0.99
    return targets


def normalize_to_range(values: np.ndarray) -> np.ndarray:
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    denom = max(max_val - min_val, 1e-8)
    normalized = (values - min_val) / denom
    return normalized * 0.98 + 0.01


class NeuralNetwork:
    def __init__(
        self,
        input_nodes: int,
        hidden_nodes: int,
        output_nodes: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        rng = np.random.default_rng(seed)
        self.wih = rng.normal(0.0, self.inodes ** -0.5, size=(self.hnodes, self.inodes))
        self.who = rng.normal(0.0, self.hnodes ** -0.5, size=(self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

    def train_batch(self, inputs_batch: np.ndarray, targets_batch: np.ndarray) -> tuple[np.ndarray, float]:
        inputs = np.asarray(inputs_batch, dtype=np.float32)
        targets = np.asarray(targets_batch, dtype=np.float32)

        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)

        batch_size = inputs.shape[0]
        inputs_t = inputs.T
        targets_t = targets.T

        hidden_inputs = np.dot(self.wih, inputs_t)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets_t - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        output_grad = output_errors * final_outputs * (1.0 - final_outputs)
        hidden_grad = hidden_errors * hidden_outputs * (1.0 - hidden_outputs)

        self.who += self.lr * np.dot(output_grad, hidden_outputs.T) / batch_size
        self.wih += self.lr * np.dot(hidden_grad, inputs_t.T) / batch_size

        sample_mse = float(np.mean((targets_t - final_outputs) ** 2))
        return final_outputs.T, sample_mse

    def query_batch(self, inputs_batch: np.ndarray) -> np.ndarray:
        inputs = np.asarray(inputs_batch, dtype=np.float32)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        inputs_t = inputs.T

        hidden_inputs = np.dot(self.wih, inputs_t)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs.T

    def backquery(self, targets_list: np.ndarray) -> np.ndarray:
        final_outputs = np.array(targets_list, ndmin=2).T
        final_outputs = np.clip(final_outputs, 1e-6, 1 - 1e-6)

        final_inputs = self.inverse_activation_function(final_outputs)

        hidden_outputs = np.dot(self.who.T, final_inputs)
        hidden_outputs = normalize_to_range(hidden_outputs)

        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        inputs = np.dot(self.wih.T, hidden_inputs)
        inputs = normalize_to_range(inputs)

        return inputs


def evaluate_model(
    model: NeuralNetwork,
    images: np.ndarray,
    labels: np.ndarray,
    output_nodes: int,
    batch_size: int,
) -> tuple[float, np.ndarray]:
    confusion = np.zeros((output_nodes, output_nodes), dtype=np.int64)
    preds_all: list[np.ndarray] = []

    for start in range(0, len(labels), batch_size):
        end = min(start + batch_size, len(labels))
        batch_images = scale_inputs(images[start:end])
        outputs = model.query_batch(batch_images)
        preds = np.argmax(outputs, axis=1)
        preds_all.append(preds)

    pred_labels = np.concatenate(preds_all) if preds_all else np.array([], dtype=np.int64)
    true_labels = labels.astype(np.int64)
    accuracy = float(np.mean(pred_labels == true_labels)) if len(true_labels) else 0.0
    np.add.at(confusion, (true_labels, pred_labels), 1)
    return accuracy, confusion


def build_targets(labels: np.ndarray, output_nodes: int) -> np.ndarray:
    targets = np.full((labels.shape[0], output_nodes), 0.01, dtype=np.float32)
    targets[np.arange(labels.shape[0]), labels.astype(np.int64)] = 0.99
    return targets


def train_model(
    model: NeuralNetwork,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    output_nodes: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, float]], np.ndarray]:
    history: list[dict[str, float]] = []
    final_confusion = np.zeros((output_nodes, output_nodes), dtype=np.int64)
    rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(len(train_labels))
        train_correct = 0
        train_total = len(train_labels)
        mse_sum = 0.0

        for start in range(0, len(train_labels), batch_size):
            end = min(start + batch_size, len(train_labels))
            idx = perm[start:end]

            batch_inputs = scale_inputs(train_images[idx])
            batch_labels = train_labels[idx]
            batch_targets = build_targets(batch_labels, output_nodes)

            outputs, sample_mse = model.train_batch(batch_inputs, batch_targets)
            preds = np.argmax(outputs, axis=1)

            train_correct += int(np.sum(preds == batch_labels))
            mse_sum += sample_mse * (end - start)

        train_acc = float(train_correct / train_total) if train_total else 0.0
        train_mse = float(mse_sum / train_total) if train_total else 0.0
        test_acc, final_confusion = evaluate_model(
            model,
            test_images,
            test_labels,
            output_nodes,
            batch_size,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_mse,
                "train_acc": train_acc,
                "test_acc": test_acc,
            }
        )

        print(
            f"epoch={epoch} train_mse={train_mse:.6f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    return history, final_confusion


def save_metrics_csv(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "train_acc", "test_acc"])
        for row in history:
            writer.writerow(
                [
                    int(row["epoch"]),
                    row["train_mse"],
                    row["train_acc"],
                    row["test_acc"],
                ]
            )


def save_training_curves(history: list[dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(item["epoch"]) for item in history]
    train_mse = [item["train_mse"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    test_acc = [item["test_acc"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, train_mse, marker="o")
    axes[0].set_title("Experiment 3.1: Train MSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("mse")

    axes[1].plot(epochs, train_acc, marker="o", label="train acc")
    axes[1].plot(epochs, test_acc, marker="o", label="test acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix(confusion: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Experiment 3.1: Confusion Matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(confusion.shape[1]))
    ax.set_yticks(range(confusion.shape[0]))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_backquery_grid_and_vectors(
    model: NeuralNetwork,
    output_nodes: int,
    grid_path: Path,
    vectors_csv_path: Path,
) -> None:
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    with vectors_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", *[f"pixel_{i}" for i in range(28 * 28)]])

        for digit in range(output_nodes):
            targets = np.zeros(output_nodes, dtype=np.float32) + 0.01
            targets[digit] = 0.99

            image_data = model.backquery(targets).reshape(28, 28)
            writer.writerow([digit, *image_data.reshape(-1).tolist()])

            ax = axes[digit // 5, digit % 5]
            ax.imshow(image_data, cmap="Greys", interpolation="None")
            ax.set_title(f"label={digit}")
            ax.axis("off")

    fig.suptitle("Experiment 3.1: Backquery Prototypes", fontsize=12)
    fig.tight_layout()
    fig.savefig(grid_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3.1: textbook backquery experiment (default: full MNIST IDX)."
    )
    parser.add_argument("--dataset", type=str, choices=["idx", "csv"], default="idx")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-nodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    train_images, train_labels, test_images, test_labels = build_dataset(
        project_root,
        args.dataset,
        args.train_limit,
        args.test_limit,
    )

    input_nodes = 784
    output_nodes = 10

    model = NeuralNetwork(
        input_nodes=input_nodes,
        hidden_nodes=args.hidden_nodes,
        output_nodes=output_nodes,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    history, confusion = train_model(
        model=model,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        output_nodes=output_nodes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    final_test_acc = history[-1]["test_acc"] if history else 0.0

    metrics_path = project_root / "outputs" / "logs" / "ch3_exp3_1_metrics.csv"
    vectors_csv_path = project_root / "outputs" / "logs" / "ch3_exp3_1_backquery_vectors.csv"
    curves_path = project_root / "outputs" / "figures" / "ch3_exp3_1_training_curves.png"
    confusion_path = project_root / "outputs" / "figures" / "ch3_exp3_1_confusion_matrix.png"
    backquery_grid_path = project_root / "outputs" / "figures" / "ch3_exp3_1_backquery_grid.png"

    report_curves_path = project_root / "reports" / "ch3_exp3_1_training_curves.png"
    report_confusion_path = project_root / "reports" / "ch3_exp3_1_confusion_matrix.png"
    report_backquery_grid_path = project_root / "reports" / "ch3_exp3_1_backquery_grid.png"

    save_metrics_csv(history, metrics_path)
    save_training_curves(history, curves_path)
    save_confusion_matrix(confusion, confusion_path)
    save_backquery_grid_and_vectors(model, output_nodes, backquery_grid_path, vectors_csv_path)

    report_curves_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(curves_path, report_curves_path)
    shutil.copy2(confusion_path, report_confusion_path)
    shutil.copy2(backquery_grid_path, report_backquery_grid_path)

    print("=== Experiment 3.1: Backquery ===")
    print(f"dataset={args.dataset}")
    print(f"train_size={len(train_labels)}")
    print(f"test_size={len(test_labels)}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"hidden_nodes={args.hidden_nodes}")
    print(f"learning_rate={args.learning_rate}")
    print(f"final_test_accuracy={final_test_acc:.4f}")
    print(f"saved_metrics={metrics_path}")
    print(f"saved_backquery_vectors={vectors_csv_path}")
    print(f"saved_curves={curves_path}")
    print(f"saved_confusion={confusion_path}")
    print(f"saved_backquery_grid={backquery_grid_path}")
    print(f"saved_report_curves={report_curves_path}")
    print(f"saved_report_confusion={report_confusion_path}")
    print(f"saved_report_backquery_grid={report_backquery_grid_path}")


if __name__ == "__main__":
    main()
