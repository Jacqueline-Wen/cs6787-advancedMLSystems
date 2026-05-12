"""Experiment 2: System stressors — latency and stragglers across SGD variants.

Hypothesis: Sync SGD degrades sharply under communication latency and stragglers
because every worker blocks at the barrier. Local SGD is more resilient because
synchronization is infrequent. Async SGD is most resilient to both since it has
no blocking synchronization.

Stressor sweeps (4 workers, CIFAR-10 default):
  - Comm latency: inject N seconds of sleep after each all-reduce
      values: 0.0, 0.01, 0.05, 0.1  (for sync and local; async is immune)
  - Stragglers: K of 4 workers sleep S seconds before each batch
      straggler_delay=0.05s, num_stragglers in [0, 1, 2]

Usage:
  python run_experiment2.py                        # CIFAR-10, 5 epochs (primary)
  python run_experiment2.py --dataset mnist        # MNIST, 5 epochs

Results saved to results/exp2/{dataset}/
"""

import json
import os
import subprocess
import sys
import argparse

BATCH_SIZE = 64
LR = 0.01
SEED = 42
WORKERS = 4
LOCAL_H = 10

LATENCY_VALUES = [0.0, 0.01, 0.05, 0.1]
STRAGGLER_DELAY = 0.05
STRAGGLER_COUNTS = [0, 1, 2]

EPOCHS_DEFAULT = {"mnist": 5, "cifar10": 5}


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
                        help="Training epochs (default: 5)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    if args.epochs is None:
        args.epochs = EPOCHS_DEFAULT[args.dataset]

    out_dir = f"results/exp2/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    common = [
        "--dataset", args.dataset,
        "--batch-size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--epochs", str(args.epochs),
        "--seed", str(SEED),
        "--workers", str(args.workers),
    ]

    # ---- Latency sweep ----
    print("\n" + "#" * 60)
    print("# LATENCY SWEEP")
    print("#" * 60)

    for latency in LATENCY_VALUES:
        tag = f"lat{latency:.3f}".replace(".", "p")

        run(
            [sys.executable, "run_sync.py"] + common +
            ["--comm-latency", str(latency),
             "--save-metrics", f"{out_dir}/sync_{tag}.json"],
            f"Sync SGD | latency={latency}s",
        )
        run(
            [sys.executable, "run_local_sgd.py"] + common +
            ["--sync-every-h", str(LOCAL_H), "--comm-latency", str(latency),
             "--save-metrics", f"{out_dir}/local_{tag}.json"],
            f"Local SGD H={LOCAL_H} | latency={latency}s",
        )

        # Async is immune to latency — run once as a reference
        if latency == 0.0:
            run(
                [sys.executable, "run_async.py"] + common +
                ["--save-metrics", f"{out_dir}/async_no_stressor.json"],
                "Async SGD | no stressor (latency-immune reference)",
            )

    # ---- Straggler sweep ----
    print("\n" + "#" * 60)
    print("# STRAGGLER SWEEP")
    print("#" * 60)

    for n_strag in STRAGGLER_COUNTS:
        tag = f"strag{n_strag}"

        run(
            [sys.executable, "run_sync.py"] + common +
            ["--straggler-delay", str(STRAGGLER_DELAY),
             "--num-stragglers", str(n_strag),
             "--save-metrics", f"{out_dir}/sync_{tag}.json"],
            f"Sync SGD | {n_strag}/{args.workers} stragglers @ {STRAGGLER_DELAY}s",
        )
        run(
            [sys.executable, "run_local_sgd.py"] + common +
            ["--sync-every-h", str(LOCAL_H),
             "--straggler-delay", str(STRAGGLER_DELAY),
             "--num-stragglers", str(n_strag),
             "--save-metrics", f"{out_dir}/local_{tag}.json"],
            f"Local SGD H={LOCAL_H} | {n_strag}/{args.workers} stragglers @ {STRAGGLER_DELAY}s",
        )
        run(
            [sys.executable, "run_async.py"] + common +
            ["--straggler-delay", str(STRAGGLER_DELAY),
             "--num-stragglers", str(n_strag),
             "--save-metrics", f"{out_dir}/async_{tag}.json"],
            f"Async SGD | {n_strag}/{args.workers} stragglers @ {STRAGGLER_DELAY}s",
        )

    # Summary tables
    def load(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def row(hist, label):
        if hist is None:
            return f"{label:<28}  (not found)"
        last = hist[-1]
        return (f"{label:<28} {last['val_accuracy']*100:>9.2f}% "
                f"{last['wall_clock_s']:>10.1f}s")

    print(f"\n{'='*60}")
    print(f"  Experiment 2 — Latency Sweep ({args.dataset})")
    print(f"{'='*60}")
    print(f"{'Run':<28} {'Final Acc':>10} {'Time':>10}")
    print("-" * 52)
    for latency in LATENCY_VALUES:
        tag = f"lat{latency:.3f}".replace(".", "p")
        print(row(load(f"{out_dir}/sync_{tag}.json"),  f"sync  lat={latency}s"))
        print(row(load(f"{out_dir}/local_{tag}.json"), f"local lat={latency}s"))
    print(row(load(f"{out_dir}/async_no_stressor.json"), "async (no latency)"))

    print(f"\n{'='*60}")
    print(f"  Experiment 2 — Straggler Sweep ({args.dataset})")
    print(f"{'='*60}")
    print(f"{'Run':<28} {'Final Acc':>10} {'Time':>10}")
    print("-" * 52)
    for n_strag in STRAGGLER_COUNTS:
        tag = f"strag{n_strag}"
        print(row(load(f"{out_dir}/sync_{tag}.json"),  f"sync  {n_strag} strag"))
        print(row(load(f"{out_dir}/local_{tag}.json"), f"local {n_strag} strag"))
        print(row(load(f"{out_dir}/async_{tag}.json"), f"async {n_strag} strag"))

    print(f"\nResults written to {out_dir}/")
    print(f"Run `python plot_experiments.py --experiment 2 --dataset {args.dataset}` for figures.")


if __name__ == "__main__":
    main()
