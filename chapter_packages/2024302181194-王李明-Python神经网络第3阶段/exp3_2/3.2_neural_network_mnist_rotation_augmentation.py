from __future__ import annotations

import argparse
import csv
import struct
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
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
        preds_all.append(np.argmax(outputs, axis=1))

    pred_labels = np.concatenate(preds_all) if preds_all else np.array([], dtype=np.int64)
    true_labels = labels.astype(np.int64)
    accuracy = float(np.mean(pred_labels == true_labels)) if len(true_labels) else 0.0
    np.add.at(confusion, (true_labels, pred_labels), 1)
    return accuracy, confusion


def build_targets(labels: np.ndarray, output_nodes: int) -> np.ndarray:
    targets = np.full((labels.shape[0], output_nodes), 0.01, dtype=np.float32)
    targets[np.arange(labels.shape[0]), labels.astype(np.int64)] = 0.99
    return targets


def rotate_batch(flat_inputs: np.ndarray, angle: float) -> np.ndarray:
    imgs = flat_inputs.reshape(-1, 28, 28)
    rotated = np.stack(
        [
            scipy.ndimage.rotate(img, angle, cval=0.01, order=1, reshape=False)
            for img in imgs
        ],
        axis=0,
    )
    return rotated.reshape(-1, 784).astype(np.float32)


def train_model(
    model: NeuralNetwork,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    output_nodes: int,
    epochs: int,
    use_rotation: bool,
    rotation_angle: float,
    mode_name: str,
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

            inputs = scale_inputs(train_images[idx])
            labels = train_labels[idx]
            targets = build_targets(labels, output_nodes)

            outputs, sample_mse = model.train_batch(inputs, targets)
            preds = np.argmax(outputs, axis=1)

            train_correct += int(np.sum(preds == labels))
            mse_sum += sample_mse * (end - start)

            if use_rotation:
                rotated_plus = rotate_batch(inputs, rotation_angle)
                rotated_minus = rotate_batch(inputs, -rotation_angle)
                model.train_batch(rotated_plus, targets)
                model.train_batch(rotated_minus, targets)

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
            f"[{mode_name}] epoch={epoch} train_mse={train_mse:.6f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )

    return history, final_confusion


def save_comparison_metrics_csv(
    base_history: list[dict[str, float]],
    rot_history: list[dict[str, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "base_train_mse",
                "base_train_acc",
                "base_test_acc",
                "rot_train_mse",
                "rot_train_acc",
                "rot_test_acc",
            ]
        )
        for base_row, rot_row in zip(base_history, rot_history):
            writer.writerow(
                [
                    int(base_row["epoch"]),
                    base_row["train_mse"],
                    base_row["train_acc"],
                    base_row["test_acc"],
                    rot_row["train_mse"],
                    rot_row["train_acc"],
                    rot_row["test_acc"],
                ]
            )


def save_summary_csv(
    base_final_test_acc: float,
    rot_final_test_acc: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "final_test_acc"])
        writer.writerow(["baseline", base_final_test_acc])
        writer.writerow(["rotation_aug", rot_final_test_acc])
        writer.writerow(["improvement", rot_final_test_acc - base_final_test_acc])


def save_accuracy_comparison_figure(
    base_history: list[dict[str, float]],
    rot_history: list[dict[str, float]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(item["epoch"]) for item in base_history]

    base_train_acc = [item["train_acc"] for item in base_history]
    base_test_acc = [item["test_acc"] for item in base_history]

    rot_train_acc = [item["train_acc"] for item in rot_history]
    rot_test_acc = [item["test_acc"] for item in rot_history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(epochs, base_test_acc, marker="o", label="baseline test")
    axes[0].plot(epochs, rot_test_acc, marker="o", label="rotation test")
    axes[0].set_title("Experiment 3.2: Test Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend()

    axes[1].plot(epochs, base_train_acc, marker="o", label="baseline train")
    axes[1].plot(epochs, rot_train_acc, marker="o", label="rotation train")
    axes[1].set_title("Train Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_confusion_comparison_figure(
    base_confusion: np.ndarray,
    rot_confusion: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    im0 = axes[0].imshow(base_confusion, cmap="Blues")
    axes[0].set_title("Baseline Confusion")
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")
    axes[0].set_xticks(range(base_confusion.shape[1]))
    axes[0].set_yticks(range(base_confusion.shape[0]))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rot_confusion, cmap="Greens")
    axes[1].set_title("Rotation Aug Confusion")
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    axes[1].set_xticks(range(rot_confusion.shape[1]))
    axes[1].set_yticks(range(rot_confusion.shape[0]))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 3.2: compare baseline and rotation augmentation (default: full MNIST IDX)."
    )
    parser.add_argument("--dataset", type=str, choices=["idx", "csv"], default="idx")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-nodes", type=int, default=200)
    parser.add_argument("--rotation-angle", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=256)
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

    base_model = NeuralNetwork(
        input_nodes=input_nodes,
        hidden_nodes=args.hidden_nodes,
        output_nodes=output_nodes,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    rot_model = NeuralNetwork(
        input_nodes=input_nodes,
        hidden_nodes=args.hidden_nodes,
        output_nodes=output_nodes,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    base_history, base_confusion = train_model(
        model=base_model,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        output_nodes=output_nodes,
        epochs=args.epochs,
        use_rotation=False,
        rotation_angle=args.rotation_angle,
        mode_name="baseline",
        batch_size=args.batch_size,
        seed=args.seed,
    )
    rot_history, rot_confusion = train_model(
        model=rot_model,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        output_nodes=output_nodes,
        epochs=args.epochs,
        use_rotation=True,
        rotation_angle=args.rotation_angle,
        mode_name="rotation",
        batch_size=args.batch_size,
        seed=args.seed,
    )

    base_final_test_acc = base_history[-1]["test_acc"] if base_history else 0.0
    rot_final_test_acc = rot_history[-1]["test_acc"] if rot_history else 0.0
    improvement = rot_final_test_acc - base_final_test_acc

    comparison_metrics_path = project_root / "outputs" / "logs" / "ch3_exp3_2_comparison_metrics.csv"
    summary_path = project_root / "outputs" / "logs" / "ch3_exp3_2_summary.csv"
    accuracy_comparison_path = (
        project_root / "outputs" / "figures" / "ch3_exp3_2_accuracy_comparison.png"
    )
    confusion_comparison_path = (
        project_root / "outputs" / "figures" / "ch3_exp3_2_confusion_comparison.png"
    )

    report_accuracy_comparison_path = (
        project_root / "reports" / "ch3_exp3_2_accuracy_comparison.png"
    )
    report_confusion_comparison_path = (
        project_root / "reports" / "ch3_exp3_2_confusion_comparison.png"
    )

    save_comparison_metrics_csv(base_history, rot_history, comparison_metrics_path)
    save_summary_csv(base_final_test_acc, rot_final_test_acc, summary_path)
    save_accuracy_comparison_figure(base_history, rot_history, accuracy_comparison_path)
    save_confusion_comparison_figure(base_confusion, rot_confusion, confusion_comparison_path)

    report_accuracy_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(accuracy_comparison_path, report_accuracy_comparison_path)
    shutil.copy2(confusion_comparison_path, report_confusion_comparison_path)

    print("=== Experiment 3.2: Rotation Augmentation Comparison ===")
    print(f"dataset={args.dataset}")
    print(f"train_size={len(train_labels)}")
    print(f"test_size={len(test_labels)}")
    print(f"epochs={args.epochs}")
    print(f"batch_size={args.batch_size}")
    print(f"hidden_nodes={args.hidden_nodes}")
    print(f"learning_rate={args.learning_rate}")
    print(f"rotation_angle={args.rotation_angle}")
    print(f"baseline_final_test_accuracy={base_final_test_acc:.4f}")
    print(f"rotation_final_test_accuracy={rot_final_test_acc:.4f}")
    print(f"test_accuracy_improvement={improvement:.4f}")
    print(f"saved_metrics={comparison_metrics_path}")
    print(f"saved_summary={summary_path}")
    print(f"saved_accuracy_comparison={accuracy_comparison_path}")
    print(f"saved_confusion_comparison={confusion_comparison_path}")
    print(f"saved_report_accuracy_comparison={report_accuracy_comparison_path}")
    print(f"saved_report_confusion_comparison={report_confusion_comparison_path}")


if __name__ == "__main__":
    main()
