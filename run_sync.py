import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.cnn import MNISTNet, CIFAR10Net
from data.datasets import get_mnist, get_cifar10
from trainers.sync_sgd import train_sync


def worker(rank, world_size, args):
    """Entry point for each spawned worker process."""
    # Initialize the process group (gloo backend works on CPU and macOS)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    # Reproducibility — offset seed by rank so each worker shuffles differently
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    device = torch.device("cpu")

    # Partition the training data: each worker gets a disjoint slice
    if args.dataset == "mnist":
        total_samples = 60000
        indices = list(range(total_samples))
        per_worker = total_samples // world_size
        start = rank * per_worker
        end = start + per_worker if rank < world_size - 1 else total_samples
        train_loader, test_loader = get_mnist(
            args.batch_size, subset_indices=indices[start:end]
        )
        model = MNISTNet()
    else:
        total_samples = 50000
        indices = list(range(total_samples))
        per_worker = total_samples // world_size
        start = rank * per_worker
        end = start + per_worker if rank < world_size - 1 else total_samples
        train_loader, test_loader = get_cifar10(
            args.batch_size, subset_indices=indices[start:end]
        )
        model = CIFAR10Net()

    # Ensure all workers start with identical model weights
    torch.manual_seed(args.seed)
    model = MNISTNet() if args.dataset == "mnist" else CIFAR10Net()

    logger = train_sync(
        rank, world_size, model, train_loader, test_loader,
        args.lr, args.epochs, device
    )

    if rank == 0 and args.save_metrics:
        logger.save(args.save_metrics)

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Synchronous distributed SGD")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-metrics", type=str, default=None)
    args = parser.parse_args()

    print(f"Launching {args.workers} workers | Dataset: {args.dataset} | "
          f"Batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")

    mp.spawn(worker, args=(args.workers, args), nprocs=args.workers, join=True)


if __name__ == "__main__":
    main()
