import time
import json


class MetricsLogger:
    """Tracks training loss, validation accuracy, and wall-clock time per epoch."""

    def __init__(self):
        self.history = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def log_epoch(self, epoch, train_loss, val_accuracy):
        elapsed = time.time() - self.start_time
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "wall_clock_s": round(elapsed, 2),
        }
        self.history.append(entry)
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%} | "
            f"Time: {elapsed:.1f}s"
        )

    def summary(self):
        last = self.history[-1]
        print("\n--- Training Summary ---")
        print(f"Final train loss:    {last['train_loss']:.4f}")
        print(f"Final val accuracy:  {last['val_accuracy']:.2%}")
        print(f"Total wall-clock:    {last['wall_clock_s']:.1f}s")

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Metrics saved to {path}")
