"""Generate figures for Experiment 1 (H sweep), Experiment 2 (stressor sweep),
and the scaling efficiency experiment.

Usage:
  python plot_experiments.py --experiment 1 --dataset cifar10
  python plot_experiments.py --experiment 2 --dataset cifar10
  python plot_experiments.py --experiment scaling --dataset cifar10
  python plot_experiments.py --dataset cifar10        # all three
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
WORKERS_LIST = [1, 2, 4]
LOCAL_H_SCALING = 10

TARGET_ACC = {"mnist": 0.97, "cifar10": 0.70}


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


def time_to_target(hist, threshold):
    if hist is None:
        return None
    for e in hist:
        if e["val_accuracy"] >= threshold:
            return e["wall_clock_s"]
    return None


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ── Experiment 1 plots ─────────────────────────────────────────────────────────

def plot_exp1(dataset, exp1_dir, plot_dir):
    print(f"\n[Experiment 1 — {dataset}] Generating figures …")

    baseline = load(f"{exp1_dir}/baseline.json")
    sync     = load(f"{exp1_dir}/sync.json")
    async_   = load(f"{exp1_dir}/async.json")
    async_sa = load(f"{exp1_dir}/async_sa.json")
    async_bd = load(f"{exp1_dir}/async_bounded.json")
    locals_  = {h: load(f"{exp1_dir}/local_h{h}.json") for h in H_VALUES}

    valid_h = [h for h in H_VALUES if locals_[h] is not None]
    target  = TARGET_ACC[dataset]
    cmap    = plt.cm.get_cmap("YlGn", len(valid_h) + 2)

    # ── 1a: Final accuracy vs H ──────────────────────────────────────────────
    accs = [last(locals_[h], "val_accuracy") * 100 for h in valid_h]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(valid_h, accs, "o-", color="tab:green", label="Local SGD")
    if sync:
        ax.axhline(last(sync, "val_accuracy") * 100, color="tab:orange",
                   linestyle="--", label="Sync SGD")
    if async_:
        ax.axhline(last(async_, "val_accuracy") * 100, color="tab:red",
                   linestyle="-.", label="Async SGD")
    if async_sa:
        ax.axhline(last(async_sa, "val_accuracy") * 100, color="tab:purple",
                   linestyle=":", label="Async SGD (staleness-aware)")
    if async_bd:
        ax.axhline(last(async_bd, "val_accuracy") * 100, color="tab:brown",
                   linestyle=":", label="Async SGD (bounded τ=4)")
    if baseline:
        ax.axhline(last(baseline, "val_accuracy") * 100, color="tab:blue",
                   linestyle=":", label="Baseline (1 worker)")
    ax.set_xlabel("Averaging interval H (local steps)")
    ax.set_ylabel("Final validation accuracy (%)")
    ax.set_title(f"Accuracy vs. local averaging interval ({dataset})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(valid_h)
    ax.set_xticklabels([str(h) for h in valid_h])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, f"{plot_dir}/accuracy_vs_H.png")

    # ── 1b: Total communication bytes vs H ──────────────────────────────────
    comms     = [total_comm_mb(locals_[h]) for h in valid_h]
    sync_comm = total_comm_mb(sync)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if all(c is not None for c in comms):
        ax.plot(valid_h, comms, "^-", color="tab:green", label="Local SGD")
    if sync_comm is not None:
        ax.axhline(sync_comm, color="tab:orange", linestyle="--", label="Sync SGD")
    ax.set_xlabel("Averaging interval H")
    ax.set_ylabel("Total communication (MB)")
    ax.set_title(f"Communication cost vs. local averaging interval ({dataset})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(valid_h)
    ax.set_xticklabels([str(h) for h in valid_h])
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, f"{plot_dir}/comm_vs_H.png")

    # ── 1c: Accuracy vs wall-clock time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, h in enumerate(valid_h):
        hist = locals_[h]
        if hist is None:
            continue
        ax.plot([e["wall_clock_s"] for e in hist],
                [e["val_accuracy"] * 100 for e in hist],
                "o-", color=cmap(i + 1), label=f"Local H={h}")
    for hist, color, marker, label in [
        (sync,     "tab:orange", "s", "Sync SGD"),
        (async_,   "tab:red",    "D", "Async SGD"),
        (async_sa, "tab:purple", "P", "Async (staleness-aware)"),
        (async_bd, "tab:brown",  "X", "Async (bounded τ=4)"),
        (baseline, "tab:blue",   "v", "Baseline"),
    ]:
        if hist:
            ax.plot([e["wall_clock_s"] for e in hist],
                    [e["val_accuracy"] * 100 for e in hist],
                    f"{marker}--", color=color, label=label)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Accuracy vs. wall-clock time ({dataset})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, f"{plot_dir}/accuracy_vs_time_H_sweep.png")

    # ── 1d: Accuracy vs cumulative comm bytes ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, h in enumerate(valid_h):
        hist = locals_[h]
        if hist is None or "comm_bytes" not in hist[0]:
            continue
        running, cum = 0, []
        for e in hist:
            running += e.get("comm_bytes", 0)
            cum.append(running / 1e6)
        ax.plot(cum, [e["val_accuracy"] * 100 for e in hist],
                "o-", color=cmap(i + 1), label=f"Local H={h}")
    if sync and "comm_bytes" in sync[0]:
        running, cum = 0, []
        for e in sync:
            running += e.get("comm_bytes", 0)
            cum.append(running / 1e6)
        ax.plot(cum, [e["val_accuracy"] * 100 for e in sync],
                "s--", color="tab:orange", label="Sync SGD")
    ax.set_xlabel("Cumulative communication (MB)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Accuracy vs. communication cost ({dataset})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, f"{plot_dir}/accuracy_vs_comm.png")

    # ── 1e: Time-to-target-accuracy bar chart ────────────────────────────────
    entries = []
    for label, hist in [
        ("baseline",            baseline),
        ("sync",                sync),
        ("async",               async_),
        ("async\n(staleness-aware)", async_sa),
        ("async\n(bounded τ=4)", async_bd),
        *[(f"local\nH={h}", locals_[h]) for h in valid_h],
    ]:
        t = time_to_target(hist, target)
        if t is not None:
            entries.append((label, t))

    if entries:
        labels, times = zip(*entries)
        colors = (["tab:blue"] + ["tab:orange"] + ["tab:red"] * 3 +
                  ["tab:green"] * len(valid_h))[:len(entries)]
        fig, ax = plt.subplots(figsize=(max(7, len(entries) * 0.9), 4.5))
        bars = ax.bar(range(len(entries)), times, color=colors[:len(entries)])
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, t, f"{t:.0f}s",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title(f"Time to reach {target*100:.0f}% validation accuracy ({dataset})")
        ax.grid(True, alpha=0.3, axis="y")
        save_fig(fig, f"{plot_dir}/time_to_target.png")


# ── Experiment 2 plots ─────────────────────────────────────────────────────────

def plot_exp2(dataset, exp2_dir, plot_dir):
    print(f"\n[Experiment 2 — {dataset}] Generating figures …")

    def tag_lat(lat):
        return f"lat{lat:.3f}".replace(".", "p")

    sync_accs_lat  = [last(load(f"{exp2_dir}/sync_{tag_lat(l)}.json"), "val_accuracy")
                      for l in LATENCY_VALUES]
    local_accs_lat = [last(load(f"{exp2_dir}/local_{tag_lat(l)}.json"), "val_accuracy")
                      for l in LATENCY_VALUES]
    sync_times_lat  = [last(load(f"{exp2_dir}/sync_{tag_lat(l)}.json"), "wall_clock_s")
                       for l in LATENCY_VALUES]
    local_times_lat = [last(load(f"{exp2_dir}/local_{tag_lat(l)}.json"), "wall_clock_s")
                       for l in LATENCY_VALUES]

    async_ref      = load(f"{exp2_dir}/async_no_stressor.json")
    async_acc_ref  = last(async_ref, "val_accuracy")
    async_time_ref = last(async_ref, "wall_clock_s")

    # ── 2a: Latency sweep ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, values, ylabel, title in [
        (ax1,
         [v * 100 if v else None for v in sync_accs_lat],
         "Final validation accuracy (%)",
         "Accuracy vs. communication latency"),
        (ax2,
         sync_times_lat,
         "Total wall-clock time (s)",
         "Training time vs. communication latency"),
    ]:
        ax.plot(LATENCY_VALUES,
                [v * 100 if v else None for v in sync_accs_lat] if ax is ax1 else sync_times_lat,
                "s-", color="tab:orange", label="Sync SGD")
        ax.plot(LATENCY_VALUES,
                [v * 100 if v else None for v in local_accs_lat] if ax is ax1 else local_times_lat,
                "^-", color="tab:green", label=f"Local SGD H={LOCAL_H_EXP2}")
        ref_val = (async_acc_ref * 100 if async_acc_ref and ax is ax1 else async_time_ref)
        if ref_val is not None:
            ax.axhline(ref_val, color="tab:red", linestyle="--",
                       label="Async SGD (latency-immune)")
        ax.set_xlabel("Comm latency per sync (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({dataset})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    save_fig(fig, f"{plot_dir}/latency_sweep.png")

    # ── 2b: Straggler sweep ──────────────────────────────────────────────────
    def s_acc(method, n):
        return last(load(f"{exp2_dir}/{method}_strag{n}.json"), "val_accuracy")

    def s_time(method, n):
        return last(load(f"{exp2_dir}/{method}_strag{n}.json"), "wall_clock_s")

    sync_accs_s  = [s_acc("sync",  n) for n in STRAGGLER_COUNTS]
    local_accs_s = [s_acc("local", n) for n in STRAGGLER_COUNTS]
    async_accs_s = [s_acc("async", n) for n in STRAGGLER_COUNTS]
    sync_times_s  = [s_time("sync",  n) for n in STRAGGLER_COUNTS]
    local_times_s = [s_time("local", n) for n in STRAGGLER_COUNTS]
    async_times_s = [s_time("async", n) for n in STRAGGLER_COUNTS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, sync_v, local_v, async_v, ylabel, title in [
        (ax1, sync_accs_s,  local_accs_s,  async_accs_s,
         "Final validation accuracy (%)", "Accuracy vs. number of stragglers"),
        (ax2, sync_times_s, local_times_s, async_times_s,
         "Total wall-clock time (s)",     "Training time vs. number of stragglers"),
    ]:
        def pct(v): return [x * 100 if x else None for x in v]
        vals_sync  = pct(sync_v)  if ax is ax1 else sync_v
        vals_local = pct(local_v) if ax is ax1 else local_v
        vals_async = pct(async_v) if ax is ax1 else async_v

        ax.plot(STRAGGLER_COUNTS, vals_sync,  "s-", color="tab:orange", label="Sync SGD")
        ax.plot(STRAGGLER_COUNTS, vals_local, "^-", color="tab:green",
                label=f"Local SGD H={LOCAL_H_EXP2}")
        ax.plot(STRAGGLER_COUNTS, vals_async, "D-", color="tab:red",   label="Async SGD")
        ax.set_xlabel(f"Number of straggler workers ({STRAGGLER_DELAY}s delay each)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({dataset})")
        ax.set_xticks(STRAGGLER_COUNTS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    save_fig(fig, f"{plot_dir}/straggler_sweep.png")


# ── Scaling efficiency plots ───────────────────────────────────────────────────

def plot_scaling(dataset, scaling_dir, plot_dir):
    print(f"\n[Scaling — {dataset}] Generating figures …")

    methods = [
        ("Sync SGD",                   "sync",  "tab:orange", "s"),
        (f"Local SGD H={LOCAL_H_SCALING}", "local", "tab:green",  "^"),
    ]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    for method_label, prefix, color, marker in methods:
        times, accs = [], []
        for w in WORKERS_LIST:
            hist = load(f"{scaling_dir}/{prefix}_w{w}.json")
            times.append(last(hist, "wall_clock_s"))
            accs.append(last(hist, "val_accuracy"))

        base_time = times[0] if times[0] else None
        speedups    = [base_time / t if t and base_time else None for t in times]
        efficiencies = [s / w if s else None for s, w in zip(speedups, WORKERS_LIST)]

        valid = [(w, t, s, e, a) for w, t, s, e, a
                 in zip(WORKERS_LIST, times, speedups, efficiencies, accs)
                 if t is not None]
        if not valid:
            continue
        ws, ts, ss, es, as_ = zip(*valid)

        ax1.plot(ws, ts, f"{marker}-", color=color, label=method_label)
        ax2.plot(ws, ss, f"{marker}-", color=color, label=method_label)
        ax3.plot(ws, [a * 100 for a in as_], f"{marker}-", color=color, label=method_label)

    # Ideal linear speedup reference
    ax2.plot(WORKERS_LIST, WORKERS_LIST, "k--", alpha=0.4, label="Ideal linear")

    ax1.set_xlabel("Number of workers")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title(f"Training time ({dataset})")
    ax1.set_xticks(WORKERS_LIST)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Number of workers")
    ax2.set_ylabel("Speedup")
    ax2.set_title(f"Speedup ({dataset})")
    ax2.set_xticks(WORKERS_LIST)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3.set_xlabel("Number of workers")
    ax3.set_ylabel("Final validation accuracy (%)")
    ax3.set_title(f"Accuracy vs. workers ({dataset})")
    ax3.set_xticks(WORKERS_LIST)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    save_fig(fig, f"{plot_dir}/scaling_efficiency.png")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["1", "2", "scaling"], default=None,
                        help="Which experiment to plot (default: all)")
    parser.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    args = parser.parse_args()

    exp1_dir    = f"results/exp1/{args.dataset}"
    exp2_dir    = f"results/exp2/{args.dataset}"
    scaling_dir = f"results/scaling/{args.dataset}"
    plot1_dir   = f"results/plots/{args.dataset}/exp1"
    plot2_dir   = f"results/plots/{args.dataset}/exp2"
    plot_sc_dir = f"results/plots/{args.dataset}/scaling"

    run_all = args.experiment is None
    if args.experiment in ("1", None):
        if os.path.isdir(exp1_dir):
            plot_exp1(args.dataset, exp1_dir, plot1_dir)
        else:
            print(f"[skip] Experiment 1: {exp1_dir} not found")
    if args.experiment in ("2", None):
        if os.path.isdir(exp2_dir):
            plot_exp2(args.dataset, exp2_dir, plot2_dir)
        else:
            print(f"[skip] Experiment 2: {exp2_dir} not found")
    if args.experiment in ("scaling", None):
        if os.path.isdir(scaling_dir):
            plot_scaling(args.dataset, scaling_dir, plot_sc_dir)
        else:
            print(f"[skip] Scaling: {scaling_dir} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
