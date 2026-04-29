import time

import torch
import torch.nn as nn
import torch.distributed as dist

from utils.metrics import MetricsLogger


def average_parameters(model, comm_latency=0.0):
    """All-reduce model parameters across workers by averaging, then sleep for latency."""
    world_size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size
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


def train_local_sgd(rank, world_size, model, train_loader, test_loader,
                    lr, epochs, sync_every_h, device,
                    comm_latency=0.0, straggler_delay=0.0, num_stragglers=0):
    """Local SGD with periodic parameter averaging.

    Each worker trains independently for H steps, then all workers average
    their model parameters. Communication is reduced by factor H vs sync SGD.

    Args:
        rank: This worker's rank (0-indexed).
        world_size: Total number of workers.
        model: The network to train.
        train_loader: DataLoader for this worker's data partition.
        test_loader: DataLoader for the full test set.
        lr: Learning rate.
        epochs: Number of training epochs.
        sync_every_h: Local steps between parameter averaging (H).
        device: torch device.
        comm_latency: Seconds to sleep after each parameter averaging round.
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

    num_params = sum(p.numel() for p in model.parameters())
    bytes_per_sync = num_params * 4 * 2
    is_straggler = num_stragglers > 0 and rank >= world_size - num_stragglers

    global_step = 0
    total_syncs = 0

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
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % sync_every_h == 0:
                average_parameters(model, comm_latency=comm_latency)
                total_syncs += 1
                epoch_comm_bytes += bytes_per_sync

        # Sync at epoch end to ensure consistent evaluation
        if global_step % sync_every_h != 0:
            average_parameters(model, comm_latency=comm_latency)
            total_syncs += 1
            epoch_comm_bytes += bytes_per_sync

        avg_loss_tensor = torch.tensor(running_loss / num_batches, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

        if rank == 0:
            val_acc = evaluate(model, test_loader, device)
            logger.log_epoch(epoch, avg_loss, val_acc, comm_bytes=epoch_comm_bytes)

    if rank == 0:
        print(f"\nTotal sync rounds: {total_syncs} (H={sync_every_h})")
        logger.summary()

    return logger
