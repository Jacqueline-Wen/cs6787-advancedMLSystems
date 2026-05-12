"""Scaling efficiency experiment: vary number of workers for sync and local SGD.

Measures how wall-clock time and accuracy scale with number of workers compared
to ideal linear scaling. Each worker processes an equal partition of the data.

Runs:
  - Sync SGD with workers in [1, 2, 4]
  - Local SGD (H=10) with workers in [1, 2, 4]

Speedup   = time_1_worker / time_N_workers
Efficiency = speedup / N  (1.0 = perfect linear scaling)

Usage:
  python run_experiment_scaling.py                   # CIFAR-10, 15 epochs (primary)
  python run_experiment_scaling.py --dataset mnist   # MNIST, 10 epochs

Results saved to results/scaling/{dataset}/
"""

import argparse
import json
import os
import subprocess
import sys

BATCH_SIZE = 64
LR = 0.01
SEED = 42
LOCAL_H = 10
WORKERS_LIST = [1, 2, 4]
EPOCHS_DEFAULT = {"mnist": 10, "cifar10": 15}


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
                        help="Training epochs (default: 10 for mnist, 15 for cifar10)")
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = EPOCHS_DEFAULT[args.dataset]

    out_dir = f"results/scaling/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    common = [
        "--dataset", args.dataset,
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--epochs", str(args.epochs),
        "--seed", str(SEED),
    ]

    for w in WORKERS_LIST:
        run(
            [sys.executable, "run_sync.py"] + common +
            ["--workers", str(w), "--save-metrics", f"{out_dir}/sync_w{w}.json"],
            f"Sync SGD ({w} worker{'s' if w > 1 else ''})",
        )
        run(
            [sys.executable, "run_local_sgd.py"] + common +
            ["--workers", str(w), "--sync-every-h", str(LOCAL_H),
             "--save-metrics", f"{out_dir}/local_w{w}.json"],
            f"Local SGD H={LOCAL_H} ({w} worker{'s' if w > 1 else ''})",
        )

    # Summary table with speedup and scaling efficiency
    print(f"\n{'='*65}")
    print(f"  Scaling Efficiency Summary ({args.dataset}, {args.epochs} epochs)")
    print(f"{'='*65}")

    for method_label, prefix in [("Sync SGD", "sync"), (f"Local SGD H={LOCAL_H}", "local")]:
        print(f"\n  {method_label}")
        print(f"  {'Workers':>8} {'Acc':>8} {'Time (s)':>10} {'Speedup':>10} {'Efficiency':>12}")
        print("  " + "-" * 52)

        base_time = None
        for w in WORKERS_LIST:
            path = f"{out_dir}/{prefix}_w{w}.json"
            if not os.path.exists(path):
                print(f"  {w:>8}  (not found)")
                continue
            with open(path) as f:
                hist = json.load(f)
            t = hist[-1]["wall_clock_s"]
            acc = hist[-1]["val_accuracy"] * 100
            if w == 1:
                base_time = t
            speedup = base_time / t if base_time else float("nan")
            efficiency = speedup / w
            print(f"  {w:>8} {acc:>7.2f}% {t:>10.1f} {speedup:>9.2f}x {efficiency:>11.2f}")

    print(f"\nResults written to {out_dir}/")
    print(f"Run `python plot_experiments.py --experiment scaling --dataset {args.dataset}` for figures.")


if __name__ == "__main__":
    main()
