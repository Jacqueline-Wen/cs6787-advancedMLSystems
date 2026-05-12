"""Experiment 1: Sweep local averaging interval H for local SGD.

Hypothesis: More frequent averaging (small H) preserves accuracy at the cost of
more communication. Larger H reduces communication bytes but may hurt convergence.

Runs:
  - baseline (single worker, no communication)
  - sync SGD
  - async SGD (standard, staleness-aware, bounded staleness)
  - local SGD with H in [1, 2, 4, 8, 16, 32]

Usage:
  python run_experiment1.py                        # CIFAR-10, 25 epochs (primary)
  python run_experiment1.py --dataset mnist        # MNIST, 10 epochs

Results saved to results/exp1/{dataset}/
"""

import argparse
import json
import os
import subprocess
import sys

BATCH_SIZE = 64
LR = 0.01
SEED = 42
WORKERS = 4
H_VALUES = [1, 2, 4, 8, 16, 32]
EPOCHS_DEFAULT = {"mnist": 10, "cifar10": 25}
# Accuracy threshold for time-to-target metric
TARGET_ACC = {"mnist": 0.97, "cifar10": 0.70}


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  cmd: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 10 for mnist, 25 for cifar10)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = EPOCHS_DEFAULT[args.dataset]

    out_dir = f"results/exp1/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    common = [
        "--dataset", args.dataset,
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--epochs", str(args.epochs),
        "--seed", str(SEED),
    ]

    # Baseline (single worker)
    run(
        [sys.executable, "run_baseline.py"] + common +
        ["--save-metrics", f"{out_dir}/baseline.json"],
        "Baseline: single-worker SGD",
    )

    # Sync SGD
    run(
        [sys.executable, "run_sync.py"] + common +
        ["--workers", str(args.workers), "--save-metrics", f"{out_dir}/sync.json"],
        f"Sync SGD ({args.workers} workers)",
    )

    # Async SGD variants — run all three to compare staleness strategies on CIFAR-10
    run(
        [sys.executable, "run_async.py"] + common +
        ["--workers", str(args.workers), "--save-metrics", f"{out_dir}/async.json"],
        f"Async SGD ({args.workers} workers)",
    )
    run(
        [sys.executable, "run_async.py"] + common +
        ["--workers", str(args.workers), "--staleness-aware",
         "--save-metrics", f"{out_dir}/async_sa.json"],
        f"Async SGD staleness-aware ({args.workers} workers)",
    )
    run(
        [sys.executable, "run_async.py"] + common +
        ["--workers", str(args.workers), "--max-staleness", "4",
         "--save-metrics", f"{out_dir}/async_bounded.json"],
        f"Async SGD bounded staleness=4 ({args.workers} workers)",
    )

    # Local SGD sweeping H
    for h in H_VALUES:
        run(
            [sys.executable, "run_local_sgd.py"] + common +
            ["--workers", str(args.workers), "--sync-every-h", str(h),
             "--save-metrics", f"{out_dir}/local_h{h}.json"],
            f"Local SGD H={h} ({args.workers} workers)",
        )

    # Summary table
    target = TARGET_ACC[args.dataset]
    print(f"\n{'='*72}")
    print(f"  Experiment 1 Summary ({args.dataset}, {args.epochs} epochs)")
    print(f"{'='*72}")
    print(f"{'Run':<26} {'Final Acc':>10} {'Time (s)':>10} "
          f"{'Comm (MB)':>11} {'→{:.0f}% (s)'.format(target*100):>12}")
    print("-" * 72)

    def summarize(path, label):
        if not os.path.exists(path):
            return
        with open(path) as f:
            hist = json.load(f)
        last = hist[-1]
        total_comm = sum(e.get("comm_bytes", 0) for e in hist) / 1e6
        comm_str = f"{total_comm:>11.1f}" if total_comm > 0 else f"{'N/A':>11}"
        t2a = next((e["wall_clock_s"] for e in hist if e["val_accuracy"] >= target), None)
        t2a_str = f"{t2a:>12.1f}" if t2a is not None else f"{'—':>12}"
        print(f"{label:<26} {last['val_accuracy']*100:>9.2f}% "
              f"{last['wall_clock_s']:>10.1f} {comm_str} {t2a_str}")

    summarize(f"{out_dir}/baseline.json",      "baseline")
    summarize(f"{out_dir}/sync.json",           "sync")
    summarize(f"{out_dir}/async.json",          "async")
    summarize(f"{out_dir}/async_sa.json",       "async (staleness-aware)")
    summarize(f"{out_dir}/async_bounded.json",  "async (bounded τ=4)")
    for h in H_VALUES:
        summarize(f"{out_dir}/local_h{h}.json", f"local H={h}")

    print(f"\nResults written to {out_dir}/")
    print(f"Run `python plot_experiments.py --experiment 1 --dataset {args.dataset}` for figures.")


if __name__ == "__main__":
    main()
