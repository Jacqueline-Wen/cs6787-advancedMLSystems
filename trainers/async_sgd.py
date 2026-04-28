import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils.metrics import MetricsLogger


class ParameterServer:
    """Shared-memory parameter server for asynchronous SGD.

    Holds a copy of the global model parameters in shared memory.
    Workers pull (read) and push (apply gradients) through this object.
    """

    def __init__(self, model):
        # Store flattened parameter shapes and shared tensors
        self.param_shapes = []
        self.shared_params = []
        for param in model.parameters():
            self.param_shapes.append(param.shape)
            shared = param.data.clone().share_memory_()
            self.shared_params.append(shared)

        self.lock = mp.Lock()
        self.global_step = mp.Value("i", 0)

    def pull(self, model):
        """Copy global parameters into the worker's local model."""
        with self.lock:
            step = self.global_step.value
            for local_param, shared_param in zip(model.parameters(), self.shared_params):
                local_param.data.copy_(shared_param)
        return step

    def push(self, model, lr, staleness=0, staleness_aware=False):
        """Apply worker gradients to the global parameters.

        Args:
            model: Worker's local model (with .grad populated).
            lr: Base learning rate.
            staleness: How stale this gradient is (tau).
            staleness_aware: If True, scale LR by 1/(1+tau).
        """
        effective_lr = lr / (1 + staleness) if staleness_aware else lr
        with self.lock:
            for shared_param, local_param in zip(self.shared_params, model.parameters()):
                if local_param.grad is not None:
                    shared_param.data -= effective_lr * local_param.grad.data
            self.global_step.value += 1


def evaluate_from_server(ps, model, test_loader, device):
    """Pull latest parameters from the server and evaluate."""
    ps.pull(model)
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


def worker_loop(rank, world_size, ps, model_fn, train_loader, lr,
                steps_per_epoch, total_steps, staleness_aware, max_staleness,
                staleness_stats, loss_accumulator, loss_count):
    """Training loop for a single async worker.

    Args:
        rank: Worker rank.
        world_size: Total workers.
        ps: ParameterServer instance.
        model_fn: Callable that returns a fresh model (for this worker).
        train_loader: DataLoader for this worker's data partition.
        lr: Base learning rate.
        steps_per_epoch: Steps that define one epoch.
        total_steps: Total global steps to train.
        staleness_aware: Whether to scale LR by staleness.
        max_staleness: If > 0, block when staleness exceeds this.
        staleness_stats: Shared list [sum_staleness, max_staleness, count].
        loss_accumulator: Shared mp.Value for accumulating loss.
        loss_count: Shared mp.Value for counting batches.
    """
    device = torch.device("cpu")
    model = model_fn().to(device)
    criterion = nn.CrossEntropyLoss()

    data_iter = iter(train_loader)

    while True:
        # Check if we've reached the target steps
        current_global = ps.global_step.value
        if current_global >= total_steps:
            break

        # Pull latest parameters
        pull_step = ps.pull(model)

        # Get next batch, cycling through data
        try:
            data, target = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            data, target = next(data_iter)

        data, target = data.to(device), target.to(device)

        # Forward + backward
        model.train()
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Compute staleness
        staleness = ps.global_step.value - pull_step

        # Bounded staleness: wait if too stale
        if max_staleness > 0:
            while staleness > max_staleness:
                staleness = ps.global_step.value - pull_step
                if ps.global_step.value >= total_steps:
                    return

        # Push gradients
        ps.push(model, lr, staleness=staleness, staleness_aware=staleness_aware)

        # Track staleness statistics
        with staleness_stats.get_lock():
            staleness_stats[0] += staleness
            staleness_stats[1] = max(staleness_stats[1], staleness)
            staleness_stats[2] += 1

        # Accumulate loss for logging
        with loss_accumulator.get_lock():
            loss_accumulator.value += loss.item()
        with loss_count.get_lock():
            loss_count.value += 1


def train_async(world_size, model_fn, train_loaders, test_loader, lr, epochs,
                steps_per_epoch, staleness_aware, max_staleness, device):
    """Asynchronous SGD training with a shared-memory parameter server.

    Args:
        world_size: Number of workers.
        model_fn: Callable returning a fresh model instance.
        train_loaders: List of DataLoaders, one per worker.
        test_loader: DataLoader for evaluation.
        lr: Base learning rate.
        epochs: Number of epochs.
        steps_per_epoch: Global steps per epoch.
        staleness_aware: Whether to scale LR by 1/(1+staleness).
        max_staleness: Upper bound on staleness (0 = unbounded).
        device: torch device for evaluation.

    Returns:
        MetricsLogger with per-epoch history.
    """
    # Initialize parameter server with a fresh model
    model = model_fn()
    ps = ParameterServer(model)

    total_steps = epochs * steps_per_epoch

    # Shared statistics
    # staleness_stats: [sum, max, count] as a shared array
    staleness_stats = mp.Array("d", [0.0, 0.0, 0.0])
    loss_accumulator = mp.Value("d", 0.0)
    loss_count = mp.Value("i", 0)

    logger = MetricsLogger()
    logger.start()

    # Launch workers
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=worker_loop,
            args=(rank, world_size, ps, model_fn, train_loaders[rank],
                  lr, steps_per_epoch, total_steps, staleness_aware,
                  max_staleness, staleness_stats, loss_accumulator, loss_count),
        )
        p.start()
        processes.append(p)

    # Monitor and log at epoch boundaries
    eval_model = model_fn().to(device)
    last_logged_epoch = 0

    while True:
        current_step = ps.global_step.value
        current_epoch = current_step // steps_per_epoch

        if current_epoch > last_logged_epoch and current_epoch <= epochs:
            # Log this epoch
            with loss_accumulator.get_lock():
                with loss_count.get_lock():
                    if loss_count.value > 0:
                        avg_loss = loss_accumulator.value / loss_count.value
                    else:
                        avg_loss = 0.0
                    # Reset for next epoch
                    loss_accumulator.value = 0.0
                    loss_count.value = 0

            val_acc = evaluate_from_server(ps, eval_model, test_loader, device)

            avg_staleness = (staleness_stats[0] / staleness_stats[2]
                             if staleness_stats[2] > 0 else 0.0)
            max_s = staleness_stats[1]

            logger.log_epoch(current_epoch, avg_loss, val_acc)
            print(f"         Avg staleness: {avg_staleness:.1f} | "
                  f"Max staleness: {int(max_s)}")

            last_logged_epoch = current_epoch

        if current_step >= total_steps:
            break

        # Brief sleep to avoid busy-waiting in the monitor loop
        import time
        time.sleep(0.1)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print(f"\nFinal staleness stats — "
          f"Avg: {staleness_stats[0] / max(staleness_stats[2], 1):.1f} | "
          f"Max: {int(staleness_stats[1])}")
    logger.summary()
    return logger
