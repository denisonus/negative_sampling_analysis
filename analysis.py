"""Analysis and Visualization for experimental results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_results(results_file):
    with open(results_file, "r") as f:
        return json.load(f)


def plot_metric_comparison(results, metric="ndcg@10", output_path=None):
    """Create bar chart comparing strategies on a specific metric."""
    # Handle both old format and new format with statistics
    if "statistics" in results:
        stats_data = results["statistics"]
        strategies = list(stats_data.keys())
        values = [
            stats_data[s]["metrics"].get(metric, {}).get("mean", 0) for s in strategies
        ]
        errors = [
            stats_data[s]["metrics"].get(metric, {}).get("std", 0) for s in strategies
        ]
    else:
        strategies = list(results.keys())
        values = [results[s]["test_metrics"].get(metric, 0) for s in strategies]
        errors = None

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        strategies,
        values,
        color="steelblue",
        edgecolor="black",
        yerr=errors,
        capsize=5 if errors else 0,
    )
    bars[np.argmax(values)].set_color("forestgreen")

    plt.xlabel("Sampling Strategy")
    plt.ylabel(metric)
    plt.title(f"Comparison of Negative Sampling Strategies on {metric}")
    plt.xticks(rotation=45, ha="right")

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_all_metrics(
    results, metrics=["ndcg@10", "recall@10", "hit@10", "mrr@10"], output_path=None
):
    """Create grouped bar chart comparing all metrics with error bars."""
    # Handle both old format and new format with statistics
    if "statistics" in results:
        stats_data = results["statistics"]
        strategies = list(stats_data.keys())
        has_errors = True
    else:
        strategies = list(results.keys())
        stats_data = None
        has_errors = False

    n_strategies, n_metrics = len(strategies), len(metrics)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_strategies)
    width = 0.8 / n_metrics
    colors = plt.colormaps["Set2"](np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metrics):
        if stats_data:
            values = [
                stats_data[s]["metrics"].get(metric, {}).get("mean", 0)
                for s in strategies
            ]
            errors = [
                stats_data[s]["metrics"].get(metric, {}).get("std", 0)
                for s in strategies
            ]
        else:
            values = [results[s]["test_metrics"].get(metric, 0) for s in strategies]
            errors = None

        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=colors[i],
            yerr=errors if has_errors else None,
            capsize=3 if has_errors else 0,
        )

    ax.set_xlabel("Sampling Strategy")
    ax.set_ylabel("Metric Value")
    ax.set_title("Comparison of Negative Sampling Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_training_curves(results, output_path=None):
    """Plot training loss and validation metrics over epochs for each strategy."""
    if "raw_results" not in results:
        print("Training curves require raw_results data from multi-run experiments")
        return

    raw_results = results["raw_results"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(raw_results)))

    for idx, (strategy, runs) in enumerate(raw_results.items()):
        if not runs:
            continue

        # Aggregate training losses across runs
        all_losses = [run.get("train_losses", []) for run in runs]
        all_valid = [run.get("valid_metrics", []) for run in runs]

        if all_losses and all_losses[0]:
            # Find min length to align
            min_len = min(len(loss) for loss in all_losses if loss)
            losses_array = np.array([loss[:min_len] for loss in all_losses if loss])

            mean_loss = np.mean(losses_array, axis=0)
            std_loss = np.std(losses_array, axis=0)
            epochs = np.arange(len(mean_loss))

            axes[0].plot(epochs, mean_loss, label=strategy, color=colors[idx])
            axes[0].fill_between(
                epochs,
                mean_loss - std_loss,
                mean_loss + std_loss,
                alpha=0.2,
                color=colors[idx],
            )

        if all_valid and all_valid[0]:
            min_len = min(len(v) for v in all_valid if v)
            valid_array = np.array([v[:min_len] for v in all_valid if v])

            mean_valid = np.mean(valid_array, axis=0)
            std_valid = np.std(valid_array, axis=0)
            epochs = np.arange(len(mean_valid))

            axes[1].plot(epochs, mean_valid, label=strategy, color=colors[idx])
            axes[1].fill_between(
                epochs,
                mean_valid - std_valid,
                mean_valid + std_valid,
                alpha=0.2,
                color=colors[idx],
            )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Metric")
    axes[1].set_title("Validation Metric Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_timing_comparison(results, output_path=None):
    """Plot timing comparison: total time, sampling time, training time."""
    if "statistics" not in results:
        print("Timing comparison requires statistics data from multi-run experiments")
        return

    stats_data = results["statistics"]
    strategies = list(stats_data.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(strategies))
    width = 0.35

    sampling_times = [
        stats_data[s]["timing"]["sampling_time"]["mean"] for s in strategies
    ]
    training_times = [
        stats_data[s]["timing"]["training_time"]["mean"] for s in strategies
    ]

    ax.bar(x - width / 2, sampling_times, width, label="Sampling Time", color="coral")
    ax.bar(
        x + width / 2, training_times, width, label="Training Time", color="steelblue"
    )

    ax.set_xlabel("Sampling Strategy")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time Breakdown by Sampling Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_convergence_speed(
    results, target_metric="ndcg@10", threshold_percentile=0.9, output_path=None
):
    """Plot epochs to reach X% of final performance for each strategy."""
    if "raw_results" not in results:
        print("Convergence analysis requires raw_results data")
        return

    raw_results = results["raw_results"]

    convergence_epochs = {}
    for strategy, runs in raw_results.items():
        epochs_to_converge = []
        for run in runs:
            valid_metrics = run.get("valid_metrics", [])
            if not valid_metrics:
                continue

            # Use best validation metric as target (not final, since early stopping may occur)
            best_value = max(valid_metrics)
            if best_value <= 0:
                print(f"Warning: {strategy} has no positive validation metrics")
                continue

            threshold = best_value * threshold_percentile

            # Find first epoch that reaches threshold (1-based for display)
            for epoch, val in enumerate(valid_metrics, start=1):
                if val >= threshold:
                    epochs_to_converge.append(epoch)
                    break

        if epochs_to_converge:
            convergence_epochs[strategy] = {
                "mean": np.mean(epochs_to_converge),
                "std": np.std(epochs_to_converge),
                "values": epochs_to_converge,
            }

    if not convergence_epochs:
        print("No convergence data available")
        return

    strategies = list(convergence_epochs.keys())
    means = [convergence_epochs[s]["mean"] for s in strategies]
    stds = [convergence_epochs[s]["std"] for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        strategies, means, yerr=stds, capsize=5, color="teal", edgecolor="black"
    )

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Sampling Strategy")
    ax.set_ylabel(f"Epochs to {threshold_percentile * 100:.0f}% of Best Performance")
    ax.set_title("Convergence Speed Comparison")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    return convergence_epochs


def statistical_significance_test(results, metric="ndcg@10", baseline="uniform"):
    """Perform paired t-tests comparing each strategy to baseline."""
    if "statistics" not in results:
        print("Statistical tests require multi-run experiment data")
        return None

    stats_data = results["statistics"]

    if baseline not in stats_data:
        print(f"Baseline strategy '{baseline}' not found")
        return None

    baseline_values = stats_data[baseline]["metrics"].get(metric, {}).get("values", [])

    print(f"\nStatistical Significance Tests vs {baseline} on {metric}")
    print("=" * 60)

    results_table = []
    for strategy in stats_data:
        if strategy == baseline:
            continue

        strategy_values = (
            stats_data[strategy]["metrics"].get(metric, {}).get("values", [])
        )

        if len(baseline_values) != len(strategy_values) or len(baseline_values) < 2:
            print(f"{strategy}: Cannot compute (mismatched or insufficient runs)")
            continue

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(strategy_values, baseline_values)

        mean_diff = np.mean(strategy_values) - np.mean(baseline_values)
        significance = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else ""
        )

        print(
            f"{strategy:<15} diff={mean_diff:+.4f}  t={t_stat:.3f}  p={p_value:.4f} {significance}"
        )

        results_table.append(
            {
                "strategy": strategy,
                "mean_diff": mean_diff,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        )

    print("\n* p<0.05, ** p<0.01, *** p<0.001")

    return results_table


def create_latex_table(results, metrics=["ndcg@10", "recall@10", "hit@10", "mrr@10"]):
    """Create LaTeX table for paper with mean ± std."""
    # Handle both old format and new format with statistics
    if "statistics" in results:
        stats_data = results["statistics"]
        strategies = list(stats_data.keys())
        has_std = True
    else:
        strategies = list(results.keys())
        stats_data = None
        has_std = False

    # Find best values for bolding
    best_values = {}
    for metric in metrics:
        if stats_data:
            values = [
                stats_data[s]["metrics"].get(metric, {}).get("mean", 0)
                for s in strategies
            ]
        else:
            values = [results[s]["test_metrics"].get(metric, 0) for s in strategies]
        best_values[metric] = max(values)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparison of Negative Sampling Strategies}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        " & ".join(["Strategy"] + [m.upper() for m in metrics]) + " \\\\",
        "\\midrule",
    ]

    for strategy in strategies:
        row = [strategy.replace("_", "\\_")]
        for metric in metrics:
            if stats_data:
                m = stats_data[strategy]["metrics"].get(metric, {})
                mean = m.get("mean", 0)
                std = m.get("std", 0)
                value_str = f"{mean:.4f}$\\pm${std:.4f}" if has_std else f"{mean:.4f}"
            else:
                mean = results[strategy]["test_metrics"].get(metric, 0)
                value_str = f"{mean:.4f}"

            if abs(mean - best_values[metric]) < 1e-6:
                value_str = f"\\textbf{{{value_str}}}"
            row.append(value_str)
        lines.append(" & ".join(row) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def generate_full_report(results_file, output_dir="results"):
    """Generate a full analysis report with all visualizations."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)

    print("Generating analysis report...")

    # Metric comparison
    plot_all_metrics(results, output_path=os.path.join(output_dir, "comparison.png"))

    # Individual metrics
    for metric in ["ndcg@10", "recall@10", "hit@10", "mrr@10"]:
        plot_metric_comparison(
            results,
            metric=metric,
            output_path=os.path.join(output_dir, f"{metric.replace('@', '_')}.png"),
        )

    # Training curves
    if "raw_results" in results:
        plot_training_curves(
            results, output_path=os.path.join(output_dir, "training_curves.png")
        )
        plot_convergence_speed(
            results, output_path=os.path.join(output_dir, "convergence.png")
        )

    # Timing
    if "statistics" in results:
        plot_timing_comparison(
            results, output_path=os.path.join(output_dir, "timing.png")
        )
        statistical_significance_test(results)

    # LaTeX table
    latex_table = create_latex_table(results)
    with open(os.path.join(output_dir, "table.tex"), "w") as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {os.path.join(output_dir, 'table.tex')}")

    print("\nLaTeX Table:")
    print(latex_table)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
        generate_full_report(results_file, output_dir)
    else:
        print("Usage: python analysis.py <results_file.json> [output_dir]")
