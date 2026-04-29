import argparse
import random

import numpy as np
import torch
import torch.multiprocessing as mp

from models.cnn import MNISTNet, CIFAR10Net
from data.datasets import get_mnist, get_cifar10
from trainers.async_sgd import train_async


def main():
    parser = argparse.ArgumentParser(description="Asynchronous SGD with parameter server")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--staleness-aware", action="store_true",
                        help="Scale LR by 1/(1+staleness)")
    parser.add_argument("--max-staleness", type=int, default=0,
                        help="Bounded staleness threshold (0 = unbounded)")
    parser.add_argument("--straggler-delay", type=float, default=0.0,
                        help="Extra seconds to sleep per step for straggler workers")
    parser.add_argument("--num-stragglers", type=int, default=0,
                        help="Number of highest-ranked workers to treat as stragglers")
    parser.add_argument("--save-metrics", type=str, default=None)
    args = parser.parse_args()

    # Async uses spawned processes that share memory; spawn is required on macOS
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    print(f"Launching {args.workers} async workers | Dataset: {args.dataset} | "
          f"Batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs} | "
          f"Staleness-aware: {args.staleness_aware} | Max staleness: {args.max_staleness}")

    # Partition the training data across workers (disjoint slices)
    if args.dataset == "mnist":
        total_samples = 60000
        model_fn = MNISTNet
        loader_fn = get_mnist
    else:
        total_samples = 50000
        model_fn = CIFAR10Net
        loader_fn = get_cifar10

    indices = list(range(total_samples))
    per_worker = total_samples // args.workers

    train_loaders = []
    test_loader = None
    for rank in range(args.workers):
        start = rank * per_worker
        end = start + per_worker if rank < args.workers - 1 else total_samples
        tl, vl = loader_fn(args.batch_size, subset_indices=indices[start:end])
        train_loaders.append(tl)
        if rank == 0:
            test_loader = vl

    steps_per_epoch = total_samples // args.batch_size

    logger = train_async(
        world_size=args.workers,
        model_fn=model_fn,
        train_loaders=train_loaders,
        test_loader=test_loader,
        lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        staleness_aware=args.staleness_aware,
        max_staleness=args.max_staleness,
        device=device,
        straggler_delay=args.straggler_delay,
        num_stragglers=args.num_stragglers,
    )

    if args.save_metrics:
        logger.save(args.save_metrics)


if __name__ == "__main__":
    main()
