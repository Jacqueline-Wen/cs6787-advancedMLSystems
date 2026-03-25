import torch
import torch.nn as nn
import torch.distributed as dist

from utils.metrics import MetricsLogger


def average_gradients(model):
    """All-reduce gradients across workers by averaging."""
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


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


def train_sync(rank, world_size, model, train_loader, test_loader, lr, epochs, device):
    """Synchronous distributed SGD training loop for a single worker.

    Each worker computes gradients on its own data partition, then
    all-reduces (averages) gradients before the optimizer step. This
    ensures all workers maintain identical model parameters.

    Args:
        rank: This worker's rank (0-indexed).
        world_size: Total number of workers.
        model: The network to train (will be identical across workers).
        train_loader: DataLoader for this worker's data partition.
        test_loader: DataLoader for the full test set (evaluation).
        lr: Learning rate.
        epochs: Number of training epochs.
        device: torch device.

    Returns:
        MetricsLogger with per-epoch history (only meaningful on rank 0).
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger()
    logger.start()

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

            # Synchronize gradients across all workers
            average_gradients(model)

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # Aggregate loss across workers for consistent logging
        avg_loss_tensor = torch.tensor(running_loss / num_batches, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

        # Only rank 0 evaluates and logs
        if rank == 0:
            val_acc = evaluate(model, test_loader, device)
            logger.log_epoch(epoch, avg_loss, val_acc)

    if rank == 0:
        logger.summary()

    return logger
