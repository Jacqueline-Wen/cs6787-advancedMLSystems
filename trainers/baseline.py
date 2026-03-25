import torch
import torch.nn as nn

from utils.metrics import MetricsLogger


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


def train_baseline(model, train_loader, test_loader, lr, epochs, device):
    """Single-worker mini-batch SGD training loop.

    Args:
        model: The network to train.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test/validation set.
        lr: Learning rate.
        epochs: Number of training epochs.
        device: torch device (cpu or cuda).

    Returns:
        MetricsLogger with per-epoch history.
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
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        val_acc = evaluate(model, test_loader, device)
        logger.log_epoch(epoch, avg_loss, val_acc)

    logger.summary()
    return logger
