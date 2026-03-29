"""Analysis and Visualization for experimental results."""

import argparse
import csv
import os
import sys
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_results(results_file):
    with open(results_file, "r") as f:
        return json.load(f)


def _finalize_figure(fig, output_path=None):
    """Save or display a figure, then close it."""
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _sorted_strategies(stats_data, metric="ndcg@10"):
    return sorted(
        stats_data.keys(),
        key=lambda strategy: stats_data[strategy]["metrics"].get(metric, {}).get("mean", 0),
        reverse=True,
    )


def _safe_normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    min_value = values.min()
    max_value = values.max()
    if np.isclose(min_value, max_value):
        return np.full_like(values, 0.5)
    return (values - min_value) / (max_value - min_value)


def _get_metric_value(strategy_stats, metric):
    """Read a metric value from either relevance or quality metrics."""
    if metric in strategy_stats.get("metrics", {}):
        return strategy_stats["metrics"][metric]["mean"]
    if metric in strategy_stats.get("quality_metrics", {}):
        return strategy_stats["quality_metrics"][metric]["mean"]
    if metric in strategy_stats.get("timing", {}):
        return strategy_stats["timing"][metric]["mean"]
    return 0.0


def plot_metric_comparison(results, metric="ndcg@10", output_path=None):
    """Create bar chart comparing strategies on a specific metric."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data, metric=metric)
    values = [
        stats_data[s]["metrics"].get(metric, {}).get("mean", 0) for s in strategies
    ]
    errors = [
        stats_data[s]["metrics"].get(metric, {}).get("std", 0) for s in strategies
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        strategies,
        values,
        color="steelblue",
        edgecolor="black",
        yerr=errors,
        capsize=5,
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
    _finalize_figure(plt.gcf(), output_path)


def plot_all_metrics(
    results, metrics=["ndcg@10", "recall@10", "hit@10", "mrr@10"], output_path=None
):
    """Create grouped bar chart comparing all metrics with error bars."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)

    n_strategies, n_metrics = len(strategies), len(metrics)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_strategies)
    width = 0.8 / n_metrics
    colors = plt.colormaps["Set2"](np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metrics):
        values = [
            stats_data[s]["metrics"].get(metric, {}).get("mean", 0) for s in strategies
        ]
        errors = [
            stats_data[s]["metrics"].get(metric, {}).get("std", 0) for s in strategies
        ]

        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=colors[i],
            yerr=errors,
            capsize=3,
        )

    ax.set_xlabel("Sampling Strategy")
    ax.set_ylabel("Metric Value")
    ax.set_title("Comparison of Negative Sampling Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    _finalize_figure(fig, output_path)


def plot_quality_metrics(
    results,
    metrics=[
        "item_coverage@10",
        "novelty@10",
        "tail_percentage@10",
        "personalization@10",
    ],
    output_path=None,
):
    """Create grouped bar chart for recommendation-quality metrics."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)

    n_strategies, n_metrics = len(strategies), len(metrics)
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_strategies)
    width = 0.8 / n_metrics
    colors = plt.colormaps["Set3"](np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metrics):
        values = [
            stats_data[s].get("quality_metrics", {}).get(metric, {}).get("mean", 0)
            for s in strategies
        ]
        errors = [
            stats_data[s].get("quality_metrics", {}).get(metric, {}).get("std", 0)
            for s in strategies
        ]

        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=colors[i],
            yerr=errors,
            capsize=3,
        )

    ax.set_xlabel("Sampling Strategy")
    ax.set_ylabel("Metric Value")
    ax.set_title("Recommendation Quality Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    _finalize_figure(fig, output_path)


def plot_quality_tradeoff(
    results, x_metric="item_coverage@10", y_metric="ndcg@10", output_path=None
):
    """Plot ranking quality against recommendation-quality tradeoffs."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data, metric=y_metric)

    x_values = [
        stats_data[s].get("quality_metrics", {}).get(x_metric, {}).get("mean", 0)
        for s in strategies
    ]
    y_values = [
        stats_data[s].get("metrics", {}).get(y_metric, {}).get("mean", 0)
        for s in strategies
    ]
    sizes = [
        max(50, stats_data[s].get("timing", {}).get("total_time", {}).get("mean", 0) * 8)
        for s in strategies
    ]

    plt.figure(figsize=(10, 7))
    plt.scatter(x_values, y_values, s=sizes, color="steelblue", edgecolor="black")

    for strategy, x_value, y_value in zip(strategies, x_values, y_values):
        plt.text(x_value, y_value, strategy, fontsize=9, ha="left", va="bottom")

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"Tradeoff: {y_metric} vs {x_metric}")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    _finalize_figure(plt.gcf(), output_path)


def plot_quality_small_multiples(
    results,
    metrics=None,
    output_path=None,
):
    """Plot one small, readable bar chart per quality metric."""
    if metrics is None:
        metrics = [
            "item_coverage@10",
            "novelty@10",
            "tail_percentage@10",
            "personalization@10",
        ]

    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    colors = plt.colormaps["Set3"](np.linspace(0, 1, len(metrics)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [
            stats_data[s].get("quality_metrics", {}).get(metric, {}).get("mean", 0)
            for s in strategies
        ]
        errors = [
            stats_data[s].get("quality_metrics", {}).get(metric, {}).get("std", 0)
            for s in strategies
        ]

        bars = ax.bar(
            strategies,
            values,
            color=colors[idx],
            edgecolor="black",
            yerr=errors,
            capsize=4,
        )
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.2)

        value_offset = max(values) * 0.02 if values and max(values) > 0 else 0.01
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + value_offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    _finalize_figure(fig, output_path)


def plot_metric_tradeoff(
    results,
    x_metric,
    y_metric="ndcg@10",
    output_path=None,
    title=None,
):
    """Generic tradeoff scatter for thesis comparisons."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data, metric=y_metric)

    x_values = [_get_metric_value(stats_data[s], x_metric) for s in strategies]
    y_values = [_get_metric_value(stats_data[s], y_metric) for s in strategies]
    sizes = [
        max(50, stats_data[s].get("timing", {}).get("total_time", {}).get("mean", 0) * 8)
        for s in strategies
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(x_values, y_values, s=sizes, color="steelblue", edgecolor="black")

    for strategy, x_value, y_value in zip(strategies, x_values, y_values):
        ax.text(x_value, y_value, strategy, fontsize=9, ha="left", va="bottom")

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(title or f"Tradeoff: {y_metric} vs {x_metric}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _finalize_figure(fig, output_path)


def plot_ablation_delta(
    results, baseline="uniform", metric="ndcg@10", output_path=None
):
    """Plot metric deltas relative to the chosen baseline strategy."""
    stats_data = results["statistics"]
    if baseline not in stats_data:
        return None

    strategies = [s for s in _sorted_strategies(stats_data, metric=metric) if s != baseline]
    baseline_value = _get_metric_value(stats_data[baseline], metric)
    deltas = [_get_metric_value(stats_data[s], metric) - baseline_value for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["forestgreen" if value >= 0 else "indianred" for value in deltas]
    bars = ax.bar(strategies, deltas, color=colors, edgecolor="black")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"{metric} Delta vs {baseline}")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.2)

    offset = max(abs(v) for v in deltas) * 0.05 if deltas and max(abs(v) for v in deltas) > 0 else 0.002
    for bar, value in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return deltas


def _extract_valid_series(valid_history, metric="ndcg@10"):
    """Normalize validation history to a numeric series."""
    if not valid_history:
        return []
    if isinstance(valid_history[0], dict):
        return [entry.get(metric, 0.0) for entry in valid_history]
    return valid_history


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
            valid_series = [_extract_valid_series(v) for v in all_valid if v]
            min_len = min(len(v) for v in valid_series if v)
            valid_array = np.array([v[:min_len] for v in valid_series if v])

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
    _finalize_figure(fig, output_path)


def plot_timing_comparison(results, output_path=None):
    """Plot timing comparison: total time, sampling time, training time."""
    if "statistics" not in results:
        print("Timing comparison requires statistics data from multi-run experiments")
        return

    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)

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
    _finalize_figure(fig, output_path)


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
            valid_metrics = _extract_valid_series(run.get("valid_metrics", []))
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
    _finalize_figure(fig, output_path)

    return convergence_epochs


def plot_thesis_dashboard(results, output_path=None):
    """Create a compact dashboard with thesis-critical comparisons."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Relevance overview
    relevance_metrics = ["ndcg@10", "recall@10", "mrr@10", "hit@10"]
    x = np.arange(len(strategies))
    width = 0.8 / len(relevance_metrics)
    colors = plt.colormaps["Set2"](np.linspace(0, 1, len(relevance_metrics)))
    for idx, metric in enumerate(relevance_metrics):
        values = [
            stats_data[s]["metrics"].get(metric, {}).get("mean", 0) for s in strategies
        ]
        offset = (idx - len(relevance_metrics) / 2 + 0.5) * width
        axes[0, 0].bar(
            x + offset,
            values,
            width,
            color=colors[idx],
            label=metric,
        )
    axes[0, 0].set_title("Relevance Metrics (@10)")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(strategies, rotation=45, ha="right")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, axis="y", alpha=0.2)

    total_times = [
        stats_data[s]["timing"]["total_time"]["mean"] for s in strategies
    ]
    ndcg_values = [
        stats_data[s]["metrics"]["ndcg@10"]["mean"] for s in strategies
    ]
    axes[0, 1].scatter(total_times, ndcg_values, color="steelblue", edgecolor="black")
    for strategy, total_time, ndcg_value in zip(strategies, total_times, ndcg_values):
        axes[0, 1].text(
            total_time,
            ndcg_value,
            strategy,
            fontsize=9,
            ha="left",
            va="bottom",
        )
    axes[0, 1].set_title("Efficiency Frontier")
    axes[0, 1].set_xlabel("Total Time (s)")
    axes[0, 1].set_ylabel("NDCG@10")
    axes[0, 1].grid(True, alpha=0.3)

    coverage_values = [
        stats_data[s].get("quality_metrics", {}).get("item_coverage@10", {}).get("mean", 0)
        for s in strategies
    ]
    axes[1, 0].scatter(
        coverage_values, ndcg_values, color="darkorange", edgecolor="black"
    )
    for strategy, coverage_value, ndcg_value in zip(
        strategies, coverage_values, ndcg_values
    ):
        axes[1, 0].text(
            coverage_value,
            ndcg_value,
            strategy,
            fontsize=9,
            ha="left",
            va="bottom",
        )
    axes[1, 0].set_title("Relevance vs Coverage")
    axes[1, 0].set_xlabel("item_coverage@10")
    axes[1, 0].set_ylabel("NDCG@10")
    axes[1, 0].grid(True, alpha=0.3)

    # Sampling vs training cost
    sampling_times = [
        stats_data[s]["timing"]["sampling_time"]["mean"] for s in strategies
    ]
    training_times = [
        stats_data[s]["timing"]["training_time"]["mean"] for s in strategies
    ]
    axes[1, 1].barh(strategies, sampling_times, color="coral", label="Sampling")
    axes[1, 1].barh(
        strategies,
        training_times,
        left=sampling_times,
        color="steelblue",
        label="Training",
    )
    axes[1, 1].set_title("Time Breakdown")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    _finalize_figure(fig, output_path)


def plot_training_dynamics(
    results, target_metric="ndcg@10", threshold_percentile=0.9, output_path=None
):
    """Plot validation dynamics and convergence speed in one compact figure."""
    if "raw_results" not in results:
        return None

    raw_results = results["raw_results"]
    strategies = list(raw_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(strategies)))

    convergence = {}

    for idx, (strategy, runs) in enumerate(raw_results.items()):
        valid_series = [
            _extract_valid_series(run.get("valid_metrics", []), metric=target_metric)
            for run in runs
            if run.get("valid_metrics")
        ]
        valid_series = [series for series in valid_series if series]
        if not valid_series:
            continue

        min_len = min(len(series) for series in valid_series)
        valid_array = np.array([series[:min_len] for series in valid_series])
        mean_valid = np.mean(valid_array, axis=0)
        std_valid = np.std(valid_array, axis=0)
        epochs = np.arange(1, len(mean_valid) + 1)

        axes[0].plot(epochs, mean_valid, label=strategy, color=colors[idx])
        axes[0].fill_between(
            epochs,
            mean_valid - std_valid,
            mean_valid + std_valid,
            alpha=0.2,
            color=colors[idx],
        )

        epochs_to_converge = []
        for series in valid_series:
            best_value = max(series)
            if best_value <= 0:
                continue
            threshold = best_value * threshold_percentile
            for epoch, value in enumerate(series, start=1):
                if value >= threshold:
                    epochs_to_converge.append(epoch)
                    break
        if epochs_to_converge:
            convergence[strategy] = {
                "mean": float(np.mean(epochs_to_converge)),
                "std": float(np.std(epochs_to_converge)),
            }

    axes[0].set_title(f"Validation {target_metric.upper()} Dynamics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(target_metric)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    if convergence:
        conv_strategies = list(convergence.keys())
        means = [convergence[s]["mean"] for s in conv_strategies]
        stds = [convergence[s]["std"] for s in conv_strategies]
        bars = axes[1].bar(
            conv_strategies,
            means,
            yerr=stds,
            capsize=4,
            color="teal",
            edgecolor="black",
        )
        for bar, mean in zip(bars, means):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axes[1].set_title(
            f"Epochs to {threshold_percentile * 100:.0f}% of Best {target_metric.upper()}"
        )
        axes[1].set_ylabel("Epoch")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, axis="y", alpha=0.2)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return convergence


def save_summary_table(results, output_path):
    """Persist a compact strategy comparison table."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)
    rows = []

    for strategy in strategies:
        strategy_stats = stats_data[strategy]
        total_time = strategy_stats["timing"]["total_time"]["mean"]
        sampling_time = strategy_stats["timing"]["sampling_time"]["mean"]
        rows.append(
            {
                "strategy": strategy,
                "ndcg@10": strategy_stats["metrics"].get("ndcg@10", {}).get("mean", 0),
                "recall@10": strategy_stats["metrics"].get("recall@10", {}).get("mean", 0),
                "mrr@10": strategy_stats["metrics"].get("mrr@10", {}).get("mean", 0),
                "hit@10": strategy_stats["metrics"].get("hit@10", {}).get("mean", 0),
                "item_coverage@10": strategy_stats.get("quality_metrics", {})
                .get("item_coverage@10", {})
                .get("mean", 0),
                "novelty@10": strategy_stats.get("quality_metrics", {})
                .get("novelty@10", {})
                .get("mean", 0),
                "tail_percentage@10": strategy_stats.get("quality_metrics", {})
                .get("tail_percentage@10", {})
                .get("mean", 0),
                "personalization@10": strategy_stats.get("quality_metrics", {})
                .get("personalization@10", {})
                .get("mean", 0),
                "total_time": total_time,
                "sampling_time": sampling_time,
                "training_time": strategy_stats["timing"]["training_time"]["mean"],
                "sampling_share": (sampling_time / total_time) if total_time > 0 else 0.0,
            }
        )

    if not rows:
        return

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_significance_table(results, output_path, metric="ndcg@10", baseline="uniform"):
    """Persist paired significance results when multi-run data is available."""
    table = statistical_significance_test(results, metric=metric, baseline=baseline)
    if not table:
        return

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)


def _load_metadata_for_results(results_file):
    """Load sibling metadata.json if it exists."""
    metadata_path = os.path.join(os.path.dirname(results_file), "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r") as metadata_file:
        return json.load(metadata_file)


def plot_parameter_sweep(
    results_files,
    strategy,
    metric="ndcg@10",
    param=None,
    output_path=None,
    csv_path=None,
):
    """Plot a sampler knob sweep across multiple finished runs."""
    bundles = []
    for results_file in results_files:
        results = load_results(results_file)
        metadata = _load_metadata_for_results(results_file)
        bundles.append(
            {
                "results_file": results_file,
                "results": results,
                "metadata": metadata,
                "config": metadata.get("config", {}),
            }
        )

    if not bundles:
        return None

    if param is None:
        ignored_keys = {"seed", "device", "metrics", "topk", "valid_metric"}
        candidate_params = []
        all_keys = set().union(*(bundle["config"].keys() for bundle in bundles))
        for key in sorted(all_keys):
            if key in ignored_keys:
                continue
            values = {json.dumps(bundle["config"].get(key, None), sort_keys=True) for bundle in bundles}
            if len(values) > 1:
                candidate_params.append(key)
        if len(candidate_params) != 1:
            raise ValueError(
                "Could not infer a single varying parameter. Pass --sweep_param explicitly."
            )
        param = candidate_params[0]

    rows = []
    for bundle in bundles:
        stats = bundle["results"].get("statistics", {})
        if strategy not in stats:
            continue
        strategy_stats = stats[strategy]
        rows.append(
            {
                "param": bundle["config"].get(param),
                "metric": _get_metric_value(strategy_stats, metric),
                "results_file": bundle["results_file"],
            }
        )

    if not rows:
        raise ValueError(f"Strategy '{strategy}' not found in provided result files.")

    def sort_key(row):
        value = row["param"]
        if isinstance(value, bool):
            return (0, int(value))
        if isinstance(value, (int, float)):
            return (0, float(value))
        return (1, str(value))

    rows.sort(key=sort_key)

    x_labels = [str(row["param"]) for row in rows]
    x_positions = np.arange(len(rows))
    y_values = [row["metric"] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_positions, y_values, marker="o", color="steelblue", linewidth=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(param)
    ax.set_ylabel(metric)
    ax.set_title(f"{strategy}: {metric} across {param}")
    ax.grid(True, alpha=0.3)

    for x_position, row in zip(x_positions, rows):
        ax.text(
            x_position,
            row["metric"],
            f"{row['metric']:.3f}",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    _finalize_figure(fig, output_path)

    if csv_path:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


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


def generate_full_report(results_file, output_dir=None):
    """Generate the compact thesis-oriented analysis bundle."""

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"analysis_{timestamp}")

    plt.switch_backend("Agg")
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)
    save_summary_table(results, os.path.join(output_dir, "summary_metrics.csv"))
    plot_thesis_dashboard(results, output_path=os.path.join(output_dir, "dashboard.png"))
    plot_all_metrics(
        results, output_path=os.path.join(output_dir, "relevance_metrics.png")
    )
    has_quality_metrics = any(
        stats_data.get("quality_metrics") for stats_data in results["statistics"].values()
    )
    if has_quality_metrics:
        plot_quality_small_multiples(
            results, output_path=os.path.join(output_dir, "quality_metrics.png")
        )
        plot_metric_tradeoff(
            results,
            x_metric="item_coverage@10",
            y_metric="ndcg@10",
            output_path=os.path.join(output_dir, "ndcg_vs_coverage.png"),
            title="NDCG@10 vs Item Coverage@10",
        )
    plot_metric_tradeoff(
        results,
        x_metric="total_time",
        y_metric="ndcg@10",
        output_path=os.path.join(output_dir, "ndcg_vs_time.png"),
        title="NDCG@10 vs Total Time",
    )
    plot_ablation_delta(
        results,
        baseline="uniform",
        metric="ndcg@10",
        output_path=os.path.join(output_dir, "ndcg10_delta_vs_uniform.png"),
    )

    if "raw_results" in results:
        plot_training_dynamics(
            results, output_path=os.path.join(output_dir, "training_dynamics.png")
        )

    save_significance_table(
        results, os.path.join(output_dir, "significance_ndcg10.csv")
    )

    print(f"Analysis bundle saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate experiment analysis plots.")
    parser.add_argument(
        "results_files",
        nargs="+",
        help="One or more results.json files.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sweep_strategy", type=str, default=None)
    parser.add_argument("--sweep_metric", type=str, default="ndcg@10")
    parser.add_argument("--sweep_param", type=str, default=None)
    args = parser.parse_args()

    if len(args.results_files) == 1 and args.sweep_strategy is None:
        generate_full_report(args.results_files[0], args.output_dir)
    else:
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(args.results_files[0]), "sweep_analysis"
        )
        os.makedirs(output_dir, exist_ok=True)
        if args.sweep_strategy is None:
            raise SystemExit(
                "Multiple results files require --sweep_strategy to build a parameter sweep."
            )
        plt.switch_backend("Agg")
        plot_parameter_sweep(
            args.results_files,
            strategy=args.sweep_strategy,
            metric=args.sweep_metric,
            param=args.sweep_param,
            output_path=os.path.join(output_dir, "parameter_sweep.png"),
            csv_path=os.path.join(output_dir, "parameter_sweep.csv"),
        )
