import torch
import torch.nn as nn
import torch.distributed as dist

from utils.metrics import MetricsLogger


def average_parameters(model):
    """All-reduce model parameters across workers by averaging."""
    world_size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size


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
                    lr, epochs, sync_every_h, device):
    """Local SGD with periodic parameter averaging.

    Each worker trains independently for H steps, then all workers
    average their model parameters. This reduces communication by
    a factor of H compared to synchronous SGD.

    Args:
        rank: This worker's rank (0-indexed).
        world_size: Total number of workers.
        model: The network to train.
        train_loader: DataLoader for this worker's data partition.
        test_loader: DataLoader for the full test set.
        lr: Learning rate.
        epochs: Number of training epochs.
        sync_every_h: Number of local steps between parameter averaging (H).
        device: torch device.

    Returns:
        MetricsLogger with per-epoch history (only meaningful on rank 0).
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger()
    logger.start()

    global_step = 0
    total_syncs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Periodic parameter averaging
            if global_step % sync_every_h == 0:
                average_parameters(model)
                total_syncs += 1

        # Sync at epoch end to ensure consistent evaluation
        if global_step % sync_every_h != 0:
            average_parameters(model)
            total_syncs += 1

        # Aggregate loss across workers
        avg_loss_tensor = torch.tensor(running_loss / num_batches, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

        if rank == 0:
            val_acc = evaluate(model, test_loader, device)
            logger.log_epoch(epoch, avg_loss, val_acc)

    if rank == 0:
        print(f"\nTotal sync rounds: {total_syncs} (H={sync_every_h})")
        logger.summary()

    return logger
