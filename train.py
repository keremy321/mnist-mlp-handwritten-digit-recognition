import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import MLP


def parse_args():
    parser = argparse.ArgumentParser(description="MNIST MLP baseline training")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baseline"))
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def save_plots(history, output_dir):
    epochs = [row["epoch"] for row in history]
    for metric, ylabel in [("loss", "Cross-entropy loss"), ("accuracy", "Accuracy (%)")]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row[f"train_{metric}"] for row in history], label="Training")
        plt.plot(epochs, [row[f"val_{metric}"] for row in history], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"MNIST MLP - {metric.title()}")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=160)
        plt.close()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.hidden1 <= 0 or args.hidden2 <= 0:
        raise ValueError("Epochs, batch size, learning rate, and hidden sizes must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_dataset, val_dataset = random_split(
        full_train, [50_000, 10_000], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    model = MLP(args.hidden1, args.hidden2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    checkpoint_path = args.output_dir / "best_model.pt"

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_accuracy": round(train_acc * 100, 4),
            "val_accuracy": round(val_acc * 100, 4),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.2%} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.2%}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    training_seconds = time.perf_counter() - start

    # Test set is used once, after model selection based on validation loss.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"Best epoch: {best_epoch} | Test accuracy: {test_acc:.2%} | Training time: {training_seconds:.1f} s")

    save_plots(history, args.output_dir)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "hyperparameters": {
            "hidden1": args.hidden1, "hidden2": args.hidden2,
            "learning_rate": args.lr, "batch_size": args.batch_size,
            "epochs": args.epochs, "seed": args.seed,
            "optimizer": "Adam", "loss": "CrossEntropyLoss",
            "activation": "ReLU", "normalization": "(x - 0.1307) / 0.3081",
        },
        "data_split": {"train": 50_000, "validation": 10_000, "test": 10_000},
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy_percent": test_acc * 100,
        "training_time_seconds": training_seconds,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
