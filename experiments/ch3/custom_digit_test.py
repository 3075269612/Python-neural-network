from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(image_path).convert("L").resize((28, 28))
    arr = np.asarray(image, dtype=np.float32)

    if arr.mean() > 127:
        arr = 255.0 - arr

    normalized = arr / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.1307) / 0.3081
    tensor = tensor.view(1, -1)
    return tensor, arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer a custom handwritten digit image with a trained model.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    image_path = Path(args.image)
    model_path = Path(args.model) if args.model else project_root / "models" / "mnist_mlp.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTMLP().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    x, arr = preprocess_image(image_path)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred = int(torch.argmax(probs).item())
    top_probs, top_indices = torch.topk(probs, k=3)

    print(f"predicted_digit={pred}")
    print("top3=")
    for digit, prob in zip(top_indices.tolist(), top_probs.tolist()):
        print(f"  digit={digit} prob={prob:.4f}")

    figure_path = project_root / "outputs" / "figures" / "ch3_custom_digit_preview.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(arr, cmap="gray")
    ax.set_title(f"predicted={pred}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(f"saved_figure={figure_path}")


if __name__ == "__main__":
    main()
