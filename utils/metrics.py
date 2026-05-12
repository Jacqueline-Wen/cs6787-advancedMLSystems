import time
import json


class MetricsLogger:
    """Tracks training loss, validation accuracy, and wall-clock time per epoch."""

    def __init__(self):
        self.history = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def log_epoch(self, epoch, train_loss, val_accuracy, comm_bytes=None):
        elapsed = time.time() - self.start_time
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "wall_clock_s": round(elapsed, 2),
        }
        if comm_bytes is not None:
            entry["comm_bytes"] = comm_bytes
        self.history.append(entry)
        comm_str = f" | Comm: {comm_bytes / 1e6:.2f} MB" if comm_bytes is not None else ""
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%} | "
            f"Time: {elapsed:.1f}s"
            f"{comm_str}"
        )

    def summary(self):
        last = self.history[-1]
        print("\n--- Training Summary ---")
        print(f"Final train loss:    {last['train_loss']:.4f}")
        print(f"Final val accuracy:  {last['val_accuracy']:.2%}")
        print(f"Total wall-clock:    {last['wall_clock_s']:.1f}s")
        if "comm_bytes" in last:
            total_comm = sum(e.get("comm_bytes", 0) for e in self.history)
            print(f"Total comm bytes:    {total_comm / 1e6:.2f} MB")

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved to {path}")


def time_to_accuracy(history, threshold):
    """Return wall-clock seconds when history first reaches threshold val accuracy, or None."""
    for entry in history:
        if entry["val_accuracy"] >= threshold:
            return entry["wall_clock_s"]
    return None
