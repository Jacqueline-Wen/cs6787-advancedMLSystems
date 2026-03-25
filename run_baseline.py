import argparse
import random

import numpy as np
import torch

from models.cnn import MNISTNet, CIFAR10Net
from data.datasets import get_mnist, get_cifar10
from trainers.baseline import train_baseline


def main():
    parser = argparse.ArgumentParser(description="Single-worker mini-batch SGD baseline")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-metrics", type=str, default=None, help="Path to save metrics JSON")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} | Batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")

    if args.dataset == "mnist":
        model = MNISTNet()
        train_loader, test_loader = get_mnist(args.batch_size)
    else:
        model = CIFAR10Net()
        train_loader, test_loader = get_cifar10(args.batch_size)

    logger = train_baseline(model, train_loader, test_loader, args.lr, args.epochs, device)

    if args.save_metrics:
        logger.save(args.save_metrics)


if __name__ == "__main__":
    main()
