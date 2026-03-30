from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class MNISTMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset_dataset(dataset, subset_size: int | None, seed: int):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_loaders(
    data_dir: Path,
    batch_size: int,
    train_subset: int | None,
    test_subset: int | None,
    seed: int,
    use_rotation: bool,
    rotation_degree: float,
) -> tuple[DataLoader, DataLoader]:
    train_steps = []
    if use_rotation:
        train_steps.append(transforms.RandomRotation(rotation_degree))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_transform = transforms.Compose(train_steps)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=train_transform)
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=test_transform)

    train_ds = subset_dataset(train_ds, train_subset, seed)
    test_ds = subset_dataset(test_ds, test_subset, seed + 1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    progress = tqdm(loader, desc="train" if training else "eval", leave=False)
    for images, labels in progress:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_count += labels.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def append_metrics(
    log_path: Path,
    epoch: int,
    use_rotation: bool,
    degree: float,
    train_loss: float,
    train_acc: float,
    test_loss: float,
    test_acc: float,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "use_rotation",
                    "rotation_degree",
                    "train_loss",
                    "train_acc",
                    "test_loss",
                    "test_acc",
                ]
            )
        writer.writerow([epoch, use_rotation, degree, train_loss, train_acc, test_loss, test_acc])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST MLP with optional rotation augmentation.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--use-rotation", action="store_true")
    parser.add_argument("--rotation-degree", type=float, default=20.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "raw"
    model_suffix = "rot" if args.use_rotation else "base"
    model_path = project_root / "models" / f"mnist_mlp_{model_suffix}.pt"
    log_path = project_root / "outputs" / "logs" / "ch3_rotate_metrics.csv"

    train_subset = args.train_subset if args.train_subset > 0 else None
    test_subset = args.test_subset if args.test_subset > 0 else None

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_loaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        train_subset=train_subset,
        test_subset=test_subset,
        seed=args.seed,
        use_rotation=args.use_rotation,
        rotation_degree=args.rotation_degree,
    )

    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"device={device}")
    print(f"use_rotation={args.use_rotation} rotation_degree={args.rotation_degree}")
    print(f"train_size={len(train_loader.dataset)} test_size={len(test_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)

        append_metrics(
            log_path=log_path,
            epoch=epoch,
            use_rotation=args.use_rotation,
            degree=args.rotation_degree,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "use_rotation": args.use_rotation,
        "rotation_degree": args.rotation_degree,
    }
    torch.save(checkpoint, model_path)

    print(f"saved_model={model_path}")
    print(f"saved_log={log_path}")


if __name__ == "__main__":
    main()
