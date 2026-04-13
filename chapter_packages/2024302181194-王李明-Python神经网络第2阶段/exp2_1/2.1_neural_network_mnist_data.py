from __future__ import annotations

import argparse
import csv
import shutil
import struct
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

    def train(self, inputs_list: np.ndarray, targets_list: np.ndarray) -> tuple[np.ndarray, float]:
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(
            output_errors * final_outputs * (1.0 - final_outputs),
            np.transpose(hidden_outputs),
        )

        self.wih += self.lr * np.dot(
            hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
            np.transpose(inputs),
        )

        sample_mse = float(np.mean((targets - final_outputs) ** 2))
        return final_outputs, sample_mse

    def query(self, inputs_list: np.ndarray) -> np.ndarray:
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def scale_inputs(flat_image: np.ndarray) -> np.ndarray:
    return (flat_image.astype(np.float32) / 255.0 * 0.99) + 0.01


def one_hot_target(label: int, output_nodes: int) -> np.ndarray:
    targets = np.zeros(output_nodes, dtype=np.float32) + 0.01
    targets[int(label)] = 0.99
    return targets


def evaluate_model(
    model: NeuralNetwork,
    images: np.ndarray,
    labels: np.ndarray,
    output_nodes: int,
) -> tuple[float, np.ndarray]:
    confusion = np.zeros((output_nodes, output_nodes), dtype=np.int64)
    correct = 0

    for image, label in zip(images, labels):
        outputs = model.query(scale_inputs(image))
        pred = int(np.argmax(outputs))
        true = int(label)
        confusion[true, pred] += 1
        if pred == true:
            correct += 1

    accuracy = float(correct / len(labels))
    return accuracy, confusion


def train_model(
    model: NeuralNetwork,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    epochs: int,
    output_nodes: int,
) -> tuple[list[dict[str, float]], np.ndarray]:
    history: list[dict[str, float]] = []
    final_confusion = np.zeros((output_nodes, output_nodes), dtype=np.int64)

    for epoch in range(1, epochs + 1):
        train_correct = 0
        mse_sum = 0.0

        for image, label in zip(train_images, train_labels):
            inputs = scale_inputs(image)
            targets = one_hot_target(int(label), output_nodes)
            outputs, sample_mse = model.train(inputs, targets)

            pred = int(np.argmax(outputs))
            if pred == int(label):
                train_correct += 1
            mse_sum += sample_mse

        train_acc = float(train_correct / len(train_labels))
        train_mse = float(mse_sum / len(train_labels))

        test_acc, final_confusion = evaluate_model(model, test_images, test_labels, output_nodes)

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
    axes[0].set_title("Experiment 2.1: Train MSE")
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

    ax.set_title("Experiment 2.1: Confusion Matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(confusion.shape[1]))
    ax.set_yticks(range(confusion.shape[0]))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2.1: textbook 3-layer neural network on MNIST IDX data."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-nodes", type=int, default=200)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    mnist_raw_dir = project_root / "data" / "raw" / "MNIST" / "raw"

    train_images_path = mnist_raw_dir / "train-images-idx3-ubyte"
    train_labels_path = mnist_raw_dir / "train-labels-idx1-ubyte"
    test_images_path = mnist_raw_dir / "t10k-images-idx3-ubyte"
    test_labels_path = mnist_raw_dir / "t10k-labels-idx1-ubyte"

    if not train_images_path.exists() or not train_labels_path.exists():
        raise FileNotFoundError("MNIST train IDX files not found under data/raw/MNIST/raw.")
    if not test_images_path.exists() or not test_labels_path.exists():
        raise FileNotFoundError("MNIST test IDX files not found under data/raw/MNIST/raw.")

    train_images, train_labels = load_mnist_idx(train_images_path, train_labels_path)
    test_images, test_labels = load_mnist_idx(test_images_path, test_labels_path)

    if args.train_limit > 0:
        train_images = train_images[: args.train_limit]
        train_labels = train_labels[: args.train_limit]
    if args.test_limit > 0:
        test_images = test_images[: args.test_limit]
        test_labels = test_labels[: args.test_limit]

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
        epochs=args.epochs,
        output_nodes=output_nodes,
    )

    final_test_acc = history[-1]["test_acc"]

    metrics_path = project_root / "outputs" / "logs" / "ch2_exp2_1_metrics.csv"
    curves_path = project_root / "outputs" / "figures" / "ch2_exp2_1_training_curves.png"
    confusion_path = project_root / "outputs" / "figures" / "ch2_exp2_1_confusion_matrix.png"

    report_curves_path = project_root / "reports" / "ch2_exp2_1_training_curves.png"
    report_confusion_path = project_root / "reports" / "ch2_exp2_1_confusion_matrix.png"

    save_metrics_csv(history, metrics_path)
    save_training_curves(history, curves_path)
    save_confusion_matrix(confusion, confusion_path)

    report_curves_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(curves_path, report_curves_path)
    shutil.copy2(confusion_path, report_confusion_path)

    print("=== Experiment 2.1: Neural Network MNIST Training ===")
    print(f"train_size={len(train_images)}")
    print(f"test_size={len(test_images)}")
    print(f"epochs={args.epochs}")
    print(f"hidden_nodes={args.hidden_nodes}")
    print(f"learning_rate={args.learning_rate}")
    print(f"final_test_accuracy={final_test_acc:.4f}")
    print(f"saved_metrics={metrics_path}")
    print(f"saved_curves={curves_path}")
    print(f"saved_confusion={confusion_path}")
    print(f"saved_report_curves={report_curves_path}")
    print(f"saved_report_confusion={report_confusion_path}")


if __name__ == "__main__":
    main()
