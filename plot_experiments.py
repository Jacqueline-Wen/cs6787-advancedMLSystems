"""Generate figures for Experiment 1 (H sweep) and Experiment 2 (stressor sweep).

Usage:
  python plot_experiments.py --experiment 1          # Exp 1 figures only
  python plot_experiments.py --experiment 2          # Exp 2 figures only
  python plot_experiments.py                         # both experiments
"""

import argparse
import json
import os

import matplotlib.pyplot as plt

H_VALUES = [1, 2, 4, 8, 16, 32]
LATENCY_VALUES = [0.0, 0.01, 0.05, 0.1]
STRAGGLER_DELAY = 0.05
STRAGGLER_COUNTS = [0, 1, 2]
LOCAL_H_EXP2 = 10

EXP1_DIR = "results/exp1"
EXP2_DIR = "results/exp2"
PLOT1_DIR = "results/plots/exp1"
PLOT2_DIR = "results/plots/exp2"


# ── helpers ────────────────────────────────────────────────────────────────────

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def last(hist, key):
    return hist[-1][key] if hist else None


def total_comm_mb(hist):
    if hist is None:
        return None
    total = sum(e.get("comm_bytes", 0) for e in hist)
    return total / 1e6 if total > 0 else None


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ── Experiment 1 plots ─────────────────────────────────────────────────────────

def plot_exp1():
    print("\n[Experiment 1] Generating figures …")

    baseline = load(f"{EXP1_DIR}/baseline.json")
    sync = load(f"{EXP1_DIR}/sync.json")
    locals_ = {h: load(f"{EXP1_DIR}/local_h{h}.json") for h in H_VALUES}

    # ── 1a: Final accuracy vs H ──────────────────────────────────────────────
    valid_h = [h for h in H_VALUES if locals_[h] is not None]
    accs = [last(locals_[h], "val_accuracy") * 100 for h in valid_h]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(valid_h, accs, "o-", color="tab:green", label="Local SGD")
    if sync:
        ax.axhline(last(sync, "val_accuracy") * 100, color="tab:orange",
                   linestyle="--", label="Sync SGD")
    if baseline:
        ax.axhline(last(baseline, "val_accuracy") * 100, color="tab:blue",
                   linestyle=":", label="Baseline (1 worker)")
    ax.set_xlabel("Averaging interval H (local steps)")
    ax.set_ylabel("Final validation accuracy (%)")
    ax.set_title("Accuracy vs. local averaging interval")
    ax.set_xscale("log", base=2)
    ax.set_xticks(valid_h)
    ax.set_xticklabels([str(h) for h in valid_h])
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, f"{PLOT1_DIR}/accuracy_vs_H.png")

    # ── 1b: Total communication bytes vs H ──────────────────────────────────
    comms = [total_comm_mb(locals_[h]) for h in valid_h]
    sync_comm = total_comm_mb(sync)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if all(c is not None for c in comms):
        ax.plot(valid_h, comms, "^-", color="tab:green", label="Local SGD")
    if sync_comm is not None:
        ax.axhline(sync_comm, color="tab:orange", linestyle="--", label="Sync SGD")
    ax.set_xlabel("Averaging interval H")
    ax.set_ylabel("Total communication (MB)")
    ax.set_title("Communication cost vs. local averaging interval")
    ax.set_xscale("log", base=2)
    ax.set_xticks(valid_h)
    ax.set_xticklabels([str(h) for h in valid_h])
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, f"{PLOT1_DIR}/comm_vs_H.png")

    # ── 1c: Accuracy vs wall-clock time (one curve per H) ───────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("YlGn", len(valid_h) + 2)
    for i, h in enumerate(valid_h):
        hist = locals_[h]
        if hist is None:
            continue
        times = [e["wall_clock_s"] for e in hist]
        accs_curve = [e["val_accuracy"] * 100 for e in hist]
        ax.plot(times, accs_curve, "o-", color=cmap(i + 1), label=f"Local H={h}")
    if sync:
        times = [e["wall_clock_s"] for e in sync]
        accs_curve = [e["val_accuracy"] * 100 for e in sync]
        ax.plot(times, accs_curve, "s--", color="tab:orange", label="Sync SGD")
    if baseline:
        times = [e["wall_clock_s"] for e in baseline]
        accs_curve = [e["val_accuracy"] * 100 for e in baseline]
        ax.plot(times, accs_curve, "D:", color="tab:blue", label="Baseline")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Accuracy vs. wall-clock time (H sweep)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, f"{PLOT1_DIR}/accuracy_vs_time_H_sweep.png")

    # ── 1d: Accuracy vs cumulative comm bytes (comm–accuracy tradeoff) ───────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, h in enumerate(valid_h):
        hist = locals_[h]
        if hist is None or "comm_bytes" not in hist[0]:
            continue
        cum_comm = []
        running = 0
        for e in hist:
            running += e.get("comm_bytes", 0)
            cum_comm.append(running / 1e6)
        accs_curve = [e["val_accuracy"] * 100 for e in hist]
        ax.plot(cum_comm, accs_curve, "o-", color=cmap(i + 1), label=f"Local H={h}")
    if sync and "comm_bytes" in sync[0]:
        cum_comm = []
        running = 0
        for e in sync:
            running += e.get("comm_bytes", 0)
            cum_comm.append(running / 1e6)
        accs_curve = [e["val_accuracy"] * 100 for e in sync]
        ax.plot(cum_comm, accs_curve, "s--", color="tab:orange", label="Sync SGD")
    ax.set_xlabel("Cumulative communication (MB)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Accuracy vs. communication cost")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, f"{PLOT1_DIR}/accuracy_vs_comm.png")


# ── Experiment 2 plots ─────────────────────────────────────────────────────────

def plot_exp2():
    print("\n[Experiment 2] Generating figures …")

    def tag_lat(lat):
        return f"lat{lat:.3f}".replace(".", "p")

    # ── 2a: Final accuracy vs comm latency ───────────────────────────────────
    sync_accs_lat = [last(load(f"{EXP2_DIR}/sync_{tag_lat(l)}.json"), "val_accuracy") for l in LATENCY_VALUES]
    local_accs_lat = [last(load(f"{EXP2_DIR}/local_{tag_lat(l)}.json"), "val_accuracy") for l in LATENCY_VALUES]
    async_ref = load(f"{EXP2_DIR}/async_no_stressor.json")
    async_acc_ref = last(async_ref, "val_accuracy") if async_ref else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def _plot_lat(ax, values, ylabel):
        ax.plot(LATENCY_VALUES, [v * 100 if v else None for v in sync_accs_lat],
                "s-o", color="tab:orange", label="Sync SGD")
        ax.plot(LATENCY_VALUES, [v * 100 if v else None for v in local_accs_lat],
                "^-", color="tab:green", label=f"Local SGD H={LOCAL_H_EXP2}")
        if async_acc_ref is not None:
            ax.axhline(async_acc_ref * 100, color="tab:red", linestyle="--",
                       label="Async SGD (latency-immune)")
        ax.set_xlabel("Comm latency per sync (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _plot_lat(ax1, sync_accs_lat, "Final validation accuracy (%)")
    ax1.set_title("Accuracy vs. communication latency")

    sync_times_lat = [last(load(f"{EXP2_DIR}/sync_{tag_lat(l)}.json"), "wall_clock_s") for l in LATENCY_VALUES]
    local_times_lat = [last(load(f"{EXP2_DIR}/local_{tag_lat(l)}.json"), "wall_clock_s") for l in LATENCY_VALUES]
    async_time_ref = last(async_ref, "wall_clock_s") if async_ref else None

    ax2.plot(LATENCY_VALUES, sync_times_lat, "s-o", color="tab:orange", label="Sync SGD")
    ax2.plot(LATENCY_VALUES, local_times_lat, "^-", color="tab:green",
             label=f"Local SGD H={LOCAL_H_EXP2}")
    if async_time_ref is not None:
        ax2.axhline(async_time_ref, color="tab:red", linestyle="--", label="Async SGD")
    ax2.set_xlabel("Comm latency per sync (s)")
    ax2.set_ylabel("Total wall-clock time (s)")
    ax2.set_title("Training time vs. communication latency")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    save_fig(fig, f"{PLOT2_DIR}/latency_sweep.png")

    # ── 2b: Final accuracy and time vs straggler count ───────────────────────
    sync_accs_strag = [last(load(f"{EXP2_DIR}/sync_strag{n}.json"), "val_accuracy") for n in STRAGGLER_COUNTS]
    local_accs_strag = [last(load(f"{EXP2_DIR}/local_strag{n}.json"), "val_accuracy") for n in STRAGGLER_COUNTS]
    async_accs_strag = [last(load(f"{EXP2_DIR}/async_strag{n}.json"), "val_accuracy") for n in STRAGGLER_COUNTS]

    sync_times_strag = [last(load(f"{EXP2_DIR}/sync_strag{n}.json"), "wall_clock_s") for n in STRAGGLER_COUNTS]
    local_times_strag = [last(load(f"{EXP2_DIR}/local_strag{n}.json"), "wall_clock_s") for n in STRAGGLER_COUNTS]
    async_times_strag = [last(load(f"{EXP2_DIR}/async_strag{n}.json"), "wall_clock_s") for n in STRAGGLER_COUNTS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    x_labels = [str(n) for n in STRAGGLER_COUNTS]

    def _plot_strag(ax, sync_vals, local_vals, async_vals, ylabel):
        ax.plot(STRAGGLER_COUNTS, [v * 100 if v else None for v in sync_vals],
                "s-o", color="tab:orange", label="Sync SGD")
        ax.plot(STRAGGLER_COUNTS, [v * 100 if v else None for v in local_vals],
                "^-", color="tab:green", label=f"Local SGD H={LOCAL_H_EXP2}")
        ax.plot(STRAGGLER_COUNTS, [v * 100 if v else None for v in async_vals],
                "D-", color="tab:red", label="Async SGD")
        ax.set_xlabel(f"Number of straggler workers (delay={STRAGGLER_DELAY}s each)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(STRAGGLER_COUNTS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    _plot_strag(ax1, sync_accs_strag, local_accs_strag, async_accs_strag,
                "Final validation accuracy (%)")
    ax1.set_title("Accuracy vs. number of stragglers")

    ax2.plot(STRAGGLER_COUNTS, sync_times_strag, "s-o", color="tab:orange", label="Sync SGD")
    ax2.plot(STRAGGLER_COUNTS, local_times_strag, "^-", color="tab:green",
             label=f"Local SGD H={LOCAL_H_EXP2}")
    ax2.plot(STRAGGLER_COUNTS, async_times_strag, "D-", color="tab:red", label="Async SGD")
    ax2.set_xlabel(f"Number of straggler workers (delay={STRAGGLER_DELAY}s each)")
    ax2.set_ylabel("Total wall-clock time (s)")
    ax2.set_title("Training time vs. number of stragglers")
    ax2.set_xticks(STRAGGLER_COUNTS)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    save_fig(fig, f"{PLOT2_DIR}/straggler_sweep.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=[1, 2], default=None,
                        help="Which experiment to plot (default: both)")
    args = parser.parse_args()

    if args.experiment in (1, None):
        plot_exp1()
    if args.experiment in (2, None):
        plot_exp2()

    print("\nDone.")


if __name__ == "__main__":
    main()
