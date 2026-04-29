"""Experiment 1: Sweep local averaging interval H for local SGD.

Hypothesis: More frequent averaging (small H) preserves accuracy at the cost of
more communication. Larger H reduces communication bytes but may hurt convergence.

Runs:
  - baseline (single worker, no communication)
  - sync SGD (4 workers, all-reduce every step)
  - local SGD with H in [1, 2, 4, 8, 16, 32] (4 workers)

Results saved to results/exp1/. Run plot_experiments.py afterward for figures.
"""

import json
import os
import subprocess
import sys

DATASET = "mnist"
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 10
SEED = 42
WORKERS = 4
H_VALUES = [1, 2, 4, 8, 16, 32]
OUT_DIR = "results/exp1"


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  cmd: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def common_args():
    return [
        "--dataset", DATASET,
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--epochs", str(EPOCHS),
        "--seed", str(SEED),
    ]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Baseline (single worker, no inter-worker communication)
    run(
        [sys.executable, "run_baseline.py"] + common_args() +
        ["--save-metrics", f"{OUT_DIR}/baseline.json"],
        "Baseline: single-worker SGD",
    )

    # Sync SGD (all-reduce every step = H=1 in terms of sync frequency)
    run(
        [sys.executable, "run_sync.py"] + common_args() +
        ["--workers", str(WORKERS), "--save-metrics", f"{OUT_DIR}/sync.json"],
        f"Sync SGD ({WORKERS} workers)",
    )

    # Local SGD sweeping H
    for h in H_VALUES:
        run(
            [sys.executable, "run_local_sgd.py"] + common_args() +
            ["--workers", str(WORKERS), "--sync-every-h", str(h),
             "--save-metrics", f"{OUT_DIR}/local_h{h}.json"],
            f"Local SGD H={h} ({WORKERS} workers)",
        )

    # Print summary table
    print(f"\n{'='*70}")
    print("  Experiment 1 Summary")
    print(f"{'='*70}")
    print(f"{'Run':<18} {'Final Acc':>10} {'Time (s)':>10} {'Total Comm (MB)':>16}")
    print("-" * 58)

    def summarize(path, label):
        if not os.path.exists(path):
            print(f"{label:<18}  (not found)")
            return
        with open(path) as f:
            hist = json.load(f)
        last = hist[-1]
        total_comm = sum(e.get("comm_bytes", 0) for e in hist) / 1e6
        comm_str = f"{total_comm:>16.1f}" if total_comm > 0 else f"{'N/A':>16}"
        print(f"{label:<18} {last['val_accuracy']*100:>9.2f}% "
              f"{last['wall_clock_s']:>10.1f} {comm_str}")

    summarize(f"{OUT_DIR}/baseline.json", "baseline")
    summarize(f"{OUT_DIR}/sync.json", "sync")
    for h in H_VALUES:
        summarize(f"{OUT_DIR}/local_h{h}.json", f"local H={h}")

    print(f"\nResults written to {OUT_DIR}/")
    print("Run `python plot_experiments.py --experiment 1` to generate figures.")


if __name__ == "__main__":
    main()
