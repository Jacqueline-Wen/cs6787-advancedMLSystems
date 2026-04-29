"""Experiment 2: System stressors — latency and stragglers across SGD variants.

Hypothesis: Sync SGD degrades sharply under communication latency and stragglers
because every worker blocks at the barrier. Local SGD is more resilient because
synchronization is infrequent. Async SGD is most resilient to both since it has
no blocking synchronization.

Stressor sweeps (each run uses 4 workers, 5 epochs, MNIST):
  - Comm latency: inject N seconds of sleep after each all-reduce
      values: 0.0, 0.01, 0.05, 0.1  (for sync and local; async is immune)
  - Stragglers: K of 4 workers sleep S seconds before each batch
      straggler_delay=0.05s, num_stragglers in [0, 1, 2]

Results saved to results/exp2/. Run plot_experiments.py afterward for figures.
"""

import json
import os
import subprocess
import sys

DATASET = "mnist"
BATCH_SIZE = 64
LR = 0.01
EPOCHS = 5          # fewer epochs to keep stressor runs tractable
SEED = 42
WORKERS = 4
LOCAL_H = 10        # fixed H for local SGD in this experiment

LATENCY_VALUES = [0.0, 0.01, 0.05, 0.1]      # seconds per all-reduce
STRAGGLER_DELAY = 0.05                          # seconds per batch for straggler workers
STRAGGLER_COUNTS = [0, 1, 2]                   # number of straggler workers (out of 4)

OUT_DIR = "results/exp2"


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
        "--workers", str(WORKERS),
    ]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Latency sweep ----
    print("\n" + "#" * 60)
    print("# LATENCY SWEEP")
    print("#" * 60)

    for latency in LATENCY_VALUES:
        tag = f"lat{latency:.3f}".replace(".", "p")

        run(
            [sys.executable, "run_sync.py"] + common_args() +
            ["--comm-latency", str(latency),
             "--save-metrics", f"{OUT_DIR}/sync_{tag}.json"],
            f"Sync SGD | latency={latency}s",
        )

        run(
            [sys.executable, "run_local_sgd.py"] + common_args() +
            ["--sync-every-h", str(LOCAL_H), "--comm-latency", str(latency),
             "--save-metrics", f"{OUT_DIR}/local_{tag}.json"],
            f"Local SGD H={LOCAL_H} | latency={latency}s",
        )

        # Async has no blocking all-reduce so latency does not apply;
        # run once (latency=0) and reuse for all latency points in plotting.
        if latency == 0.0:
            run(
                [sys.executable, "run_async.py"] + common_args() +
                ["--save-metrics", f"{OUT_DIR}/async_no_stressor.json"],
                "Async SGD | no stressor (latency-immune reference)",
            )

    # ---- Straggler sweep ----
    print("\n" + "#" * 60)
    print("# STRAGGLER SWEEP")
    print("#" * 60)

    for n_strag in STRAGGLER_COUNTS:
        tag = f"strag{n_strag}"

        run(
            [sys.executable, "run_sync.py"] + common_args() +
            ["--straggler-delay", str(STRAGGLER_DELAY),
             "--num-stragglers", str(n_strag),
             "--save-metrics", f"{OUT_DIR}/sync_{tag}.json"],
            f"Sync SGD | {n_strag}/{WORKERS} stragglers @ {STRAGGLER_DELAY}s",
        )

        run(
            [sys.executable, "run_local_sgd.py"] + common_args() +
            ["--sync-every-h", str(LOCAL_H),
             "--straggler-delay", str(STRAGGLER_DELAY),
             "--num-stragglers", str(n_strag),
             "--save-metrics", f"{OUT_DIR}/local_{tag}.json"],
            f"Local SGD H={LOCAL_H} | {n_strag}/{WORKERS} stragglers @ {STRAGGLER_DELAY}s",
        )

        run(
            [sys.executable, "run_async.py"] + common_args() +
            ["--straggler-delay", str(STRAGGLER_DELAY),
             "--num-stragglers", str(n_strag),
             "--save-metrics", f"{OUT_DIR}/async_{tag}.json"],
            f"Async SGD | {n_strag}/{WORKERS} stragglers @ {STRAGGLER_DELAY}s",
        )

    # Print summary tables
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
    print("  Experiment 2 — Latency Sweep Summary")
    print(f"{'='*60}")
    print(f"{'Run':<28} {'Final Acc':>10} {'Time':>10}")
    print("-" * 52)
    for latency in LATENCY_VALUES:
        tag = f"lat{latency:.3f}".replace(".", "p")
        print(row(load(f"{OUT_DIR}/sync_{tag}.json"),  f"sync  lat={latency}s"))
        print(row(load(f"{OUT_DIR}/local_{tag}.json"), f"local lat={latency}s"))
    print(row(load(f"{OUT_DIR}/async_no_stressor.json"), "async (no latency)"))

    print(f"\n{'='*60}")
    print("  Experiment 2 — Straggler Sweep Summary")
    print(f"{'='*60}")
    print(f"{'Run':<28} {'Final Acc':>10} {'Time':>10}")
    print("-" * 52)
    for n_strag in STRAGGLER_COUNTS:
        tag = f"strag{n_strag}"
        print(row(load(f"{OUT_DIR}/sync_{tag}.json"),  f"sync  {n_strag} strag"))
        print(row(load(f"{OUT_DIR}/local_{tag}.json"), f"local {n_strag} strag"))
        print(row(load(f"{OUT_DIR}/async_{tag}.json"), f"async {n_strag} strag"))

    print(f"\nResults written to {OUT_DIR}/")
    print("Run `python plot_experiments.py --experiment 2` to generate figures.")


if __name__ == "__main__":
    main()
