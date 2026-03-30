from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize first-layer MNIST MLP weights.")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    model_path = Path(args.model) if args.model else project_root / "models" / "mnist_mlp.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    device = torch.device("cpu")
    model = MNISTMLP().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    first_layer_weights = model.net[0].weight.detach().cpu()
    count = max(1, min(args.count, first_layer_weights.shape[0]))

    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for idx, ax in enumerate(axes):
        if idx < count:
            ax.imshow(first_layer_weights[idx].view(28, 28), cmap="seismic")
            ax.set_title(f"neuron {idx}")
        ax.axis("off")

    fig.suptitle("First Layer Weight Maps", fontsize=12)
    fig.tight_layout()

    out_path = project_root / "outputs" / "figures" / "ch3_first_layer_weights.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"visualized_count={count}")
    print(f"saved_figure={out_path}")


if __name__ == "__main__":
    main()
