from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_model import MNISTCNN


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch MNIST CNN.")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT_DIR / "checkpoints")
    parser.add_argument("--export-dir", type=Path, default=ROOT_DIR / "exports")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-export", action="store_true", help="Skip .npz weight export.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def build_loaders(
    data_dir: Path,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return running_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    test_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_accuracy": test_accuracy,
        },
        path,
    )


def export_weights_npz(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name.replace(".", "_"): tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    np.savez(path, **arrays)


def format_metrics(values: Iterable[tuple[str, float]]) -> str:
    return " ".join(f"{name}={value:.4f}" for name, value in values)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    train_loader, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_accuracy = 0.0
    if args.checkpoint is not None and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_accuracy = float(checkpoint.get("test_accuracy", 0.0))
        print(f"Loaded checkpoint from {args.checkpoint}")

    best_path = args.checkpoint_dir / "best_mnist_cnn.pt"
    last_path = args.checkpoint_dir / "last_mnist_cnn.pt"

    print(f"Training on {device} for {args.epochs} epoch(s)")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        save_checkpoint(last_path, model, optimizer, epoch, test_accuracy)
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_checkpoint(best_path, model, optimizer, epoch, test_accuracy)

        print(
            f"epoch={epoch} "
            + format_metrics(
                [
                    ("train_loss", train_loss),
                    ("train_acc", train_accuracy),
                    ("test_loss", test_loss),
                    ("test_acc", test_accuracy),
                    ("best_acc", best_accuracy),
                ]
            )
        )

    if not args.no_export:
        export_path = args.export_dir / "mnist_cnn_weights.npz"
        export_weights_npz(model, export_path)
        print(f"Exported weights to {export_path}")


if __name__ == "__main__":
    main()
