import argparse
import json
import os

import matplotlib.pyplot as plt


RUNS = [
    ("baseline", "results/baseline.json", "tab:blue", "o"),
    ("sync",     "results/sync.json",     "tab:orange", "s"),
    ("local",    "results/local.json",    "tab:green", "^"),
    ("async",    "results/async.json",    "tab:red", "D"),
]


def load_history(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot 4-way SGD comparison")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing the four metric JSONs")
    parser.add_argument("--out-dir", default="results/plots",
                        help="Where to save generated figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs = []
    for name, rel_path, color, marker in RUNS:
        path = os.path.join(args.results_dir, os.path.basename(rel_path))
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        runs.append((name, load_history(path), color, marker))

    if not runs:
        raise SystemExit("No metric files found.")

    # ---- Figure 1: validation accuracy vs. epoch ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, hist, color, marker in runs:
        epochs = [e["epoch"] for e in hist]
        accs = [e["val_accuracy"] * 100 for e in hist]
        ax.plot(epochs, accs, color=color, marker=marker, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Accuracy vs. epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "accuracy_vs_epoch.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 2: validation accuracy vs. wall-clock time ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, hist, color, marker in runs:
        times = [e["wall_clock_s"] for e in hist]
        accs = [e["val_accuracy"] * 100 for e in hist]
        ax.plot(times, accs, color=color, marker=marker, label=name)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Accuracy vs. wall-clock time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "accuracy_vs_walltime.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 3: training loss vs. epoch ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, hist, color, marker in runs:
        epochs = [e["epoch"] for e in hist]
        losses = [e["train_loss"] for e in hist]
        ax.plot(epochs, losses, color=color, marker=marker, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss vs. epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_vs_epoch.png"), dpi=150)
    plt.close(fig)

    # ---- Figure 4: bar chart of final accuracy + total wall-clock ----
    names = [r[0] for r in runs]
    final_acc = [r[1][-1]["val_accuracy"] * 100 for r in runs]
    total_time = [r[1][-1]["wall_clock_s"] for r in runs]
    colors = [r[2] for r in runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    bars1 = ax1.bar(names, final_acc, color=colors)
    ax1.set_ylabel("Final validation accuracy (%)")
    ax1.set_title("Final accuracy")
    ax1.set_ylim(min(final_acc) - 1, 100)
    for bar, v in zip(bars1, final_acc):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(names, total_time, color=colors)
    ax2.set_ylabel("Total wall-clock (s)")
    ax2.set_title("Training time")
    for bar, v in zip(bars2, total_time):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "summary_bars.png"), dpi=150)
    plt.close(fig)

    # ---- Console summary ----
    print(f"\n{'run':<10} {'final_acc':>10} {'total_time(s)':>15} {'epochs':>8}")
    for name, hist, _, _ in runs:
        last = hist[-1]
        print(f"{name:<10} {last['val_accuracy']*100:>9.2f}% "
              f"{last['wall_clock_s']:>15.1f} {last['epoch']:>8d}")
    print(f"\nFigures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
