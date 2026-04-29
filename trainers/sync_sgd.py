import time

import torch
import torch.nn as nn
import torch.distributed as dist

from utils.metrics import MetricsLogger


def average_gradients(model, comm_latency=0.0):
    """All-reduce gradients across workers by averaging, then sleep to simulate latency."""
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
    if comm_latency > 0.0:
        time.sleep(comm_latency)


def evaluate(model, test_loader, device):
    """Compute accuracy on the test set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)
    return correct / total


def train_sync(rank, world_size, model, train_loader, test_loader, lr, epochs, device,
               comm_latency=0.0, straggler_delay=0.0, num_stragglers=0):
    """Synchronous distributed SGD training loop for a single worker.

    Args:
        rank: This worker's rank (0-indexed).
        world_size: Total number of workers.
        model: The network to train (identical across workers at init).
        train_loader: DataLoader for this worker's data partition.
        test_loader: DataLoader for the full test set (evaluation).
        lr: Learning rate.
        epochs: Number of training epochs.
        device: torch device.
        comm_latency: Seconds to sleep after each all-reduce (simulates network delay).
        straggler_delay: Seconds to sleep before each batch for straggler ranks.
        num_stragglers: Number of highest-ranked workers to treat as stragglers.

    Returns:
        MetricsLogger with per-epoch history (only meaningful on rank 0).
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger()
    logger.start()

    # float32 = 4 bytes; x2 approximates bidirectional ring all-reduce traffic per worker
    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_sync = num_params * 4 * 2
    is_straggler = num_stragglers > 0 and rank >= world_size - num_stragglers

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_comm_bytes = 0

        for data, target in train_loader:
            if is_straggler:
                time.sleep(straggler_delay)

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            average_gradients(model, comm_latency=comm_latency)
            epoch_comm_bytes += bytes_per_sync

            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        # Aggregate loss across workers for consistent logging
        avg_loss_tensor = torch.tensor(running_loss / num_batches, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

        if rank == 0:
            val_acc = evaluate(model, test_loader, device)
            logger.log_epoch(epoch, avg_loss, val_acc, comm_bytes=epoch_comm_bytes)

    if rank == 0:
        logger.summary()

    return logger
