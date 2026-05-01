"""Plotting helpers for per-run analysis."""

import numpy as np
import matplotlib.pyplot as plt

from .common import (
    DEFAULT_METRIC_BY_K_BASES,
    DEFAULT_QUALITY_METRICS,
    DEFAULT_RELEVANCE_METRICS,
    _available_metric_ks,
    _available_metrics,
    _collect_bucket_labels,
    _finalize_figure,
    _get_metric_value,
    _preferred_metric,
    _sorted_strategies,
)


def _annotate_vertical_bars(ax, bars, values, fmt="{:.3f}", offset=None):
    if offset is None:
        max_abs = max((abs(value) for value in values), default=0.0)
        offset = max_abs * 0.03 if max_abs > 0 else 0.01

    annotation_positions = []
    for bar, value in zip(bars, values):
        annotation_y = value + (offset if value >= 0 else -offset)
        annotation_positions.append(annotation_y)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            annotation_y,
            fmt.format(value),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )

    _pad_y_limits_for_annotations(ax, annotation_positions)


def _pad_y_limits_for_annotations(ax, y_values, padding_fraction=0.08):
    """Reserve vertical space for labels that sit just outside plotted bars."""
    if not y_values:
        return

    bottom, top = ax.get_ylim()
    span = top - bottom
    if span <= 0:
        span = max(abs(top), abs(bottom), 1.0)

    padding = span * padding_fraction
    new_bottom = min(bottom, min(y_values) - padding)
    new_top = max(top, max(y_values) + padding)
    if new_bottom != bottom or new_top != top:
        ax.set_ylim(new_bottom, new_top)


def _label_points(ax, labels, x_values, y_values, fontsize=9):
    for label, x_value, y_value in zip(labels, x_values, y_values):
        ax.text(
            x_value,
            y_value,
            label,
            fontsize=fontsize,
            ha="left",
            va="bottom",
        )


def plot_metric_by_k(
    results,
    metric_bases=None,
    ks=(5, 10, 20, 50),
    output_path=None,
    title_suffix="",
):
    """Plot how key metrics evolve across the configured K cutoffs."""
    stats_data = results["statistics"]
    if metric_bases is None:
        metric_bases = list(DEFAULT_METRIC_BY_K_BASES)

    metric_ks = {
        metric_base: _available_metric_ks(stats_data, metric_base, ks=ks)
        for metric_base in metric_bases
    }
    metric_bases = [
        metric_base for metric_base in metric_bases if metric_ks.get(metric_base)
    ]
    if not metric_bases:
        return None

    strategies = _sorted_strategies(stats_data)
    fig, axes = plt.subplots(1, len(metric_bases), figsize=(5 * len(metric_bases), 5))
    if len(metric_bases) == 1:
        axes = [axes]
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(strategies)))

    for ax, metric_base in zip(axes, metric_bases):
        available_ks = metric_ks[metric_base]
        for color, strategy in zip(colors, strategies):
            y_values = [
                _get_metric_value(stats_data[strategy], f"{metric_base}@{k}")
                for k in available_ks
            ]
            ax.plot(
                available_ks,
                y_values,
                marker="o",
                linewidth=2,
                color=color,
                label=strategy,
            )

        ax.set_title(metric_base.upper())
        ax.set_xlabel("K")
        ax.set_ylabel("Metric value")
        ax.set_xticks(available_ks)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    fig.suptitle(f"Metric Behavior Across K{title_suffix}")
    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return metric_bases



def plot_quality_small_multiples(
    results,
    metrics=None,
    output_path=None,
    title_suffix="",
):
    """Plot one small, readable bar chart per quality metric."""
    if metrics is None:
        metrics = list(DEFAULT_QUALITY_METRICS)

    stats_data = results["statistics"]
    metrics = _available_metrics(stats_data, metrics, section="quality_metrics")
    if not metrics:
        return None

    strategies = _sorted_strategies(stats_data)
    ncols = min(2, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
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
        _annotate_vertical_bars(ax, bars, values)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(f"Recommendation Quality Metrics{title_suffix}")
    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return metrics



def plot_ablation_delta(
    results, baseline="uniform", metric="ndcg@10", output_path=None, title_suffix=""
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
    ax.set_title(f"{metric} Delta vs {baseline}{title_suffix}")
    ax.set_ylabel("Delta")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.2)
    _annotate_vertical_bars(ax, bars, deltas, fmt="{:+.3f}", offset=0.002)

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return deltas



def plot_user_bucket_delta_heatmap(
    results,
    baseline="uniform",
    metric="ndcg@10",
    output_path=None,
    title_suffix="",
):
    """Plot a compact heatmap of per-bucket deltas versus a baseline strategy."""
    stats_data = results["statistics"]
    if baseline not in stats_data:
        return None

    bucket_labels = _collect_bucket_labels(stats_data)
    if not bucket_labels:
        return None

    strategies = [
        strategy
        for strategy in _sorted_strategies(stats_data, metric=metric)
        if strategy != baseline
    ]
    if not strategies:
        return None

    delta_rows = []
    row_labels = []
    baseline_bucket_metrics = stats_data[baseline].get("bucket_metrics", {})

    for strategy in strategies:
        strategy_bucket_metrics = stats_data[strategy].get("bucket_metrics", {})
        row = []
        has_value = False
        for bucket_label in bucket_labels:
            baseline_stats = baseline_bucket_metrics.get(bucket_label, {}).get(metric)
            strategy_stats = strategy_bucket_metrics.get(bucket_label, {}).get(metric)
            if baseline_stats is None or strategy_stats is None:
                row.append(np.nan)
                continue
            has_value = True
            row.append(strategy_stats["mean"] - baseline_stats["mean"])
        if has_value:
            delta_rows.append(row)
            row_labels.append(strategy)

    if not delta_rows:
        return None

    heatmap = np.asarray(delta_rows, dtype=np.float64)
    finite_values = heatmap[np.isfinite(heatmap)]
    if finite_values.size == 0:
        return None

    max_abs = float(np.max(np.abs(finite_values)))
    if np.isclose(max_abs, 0.0):
        max_abs = 1e-6

    fig, ax = plt.subplots(
        figsize=(max(6, 1.4 * len(bucket_labels)), max(3, 0.7 * len(row_labels) + 1.5))
    )
    cmap = plt.get_cmap("RdYlGn")
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color="lightgray")
    image = ax.imshow(
        np.ma.masked_invalid(heatmap),
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(bucket_labels)))
    ax.set_xticklabels(bucket_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Train interactions")
    ax.set_title(f"{metric.upper()} Delta vs {baseline}{title_suffix}")

    for row_idx in range(heatmap.shape[0]):
        for col_idx in range(heatmap.shape[1]):
            value = heatmap[row_idx, col_idx]
            if np.isnan(value):
                ax.text(col_idx, row_idx, "NA", ha="center", va="center", fontsize=8)
                continue
            text_color = "white" if abs(value) > max_abs * 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, shrink=0.85, label="Delta")
    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return heatmap


def _extract_valid_series(valid_history, metric="ndcg@10"):
    """Normalize validation history to a numeric series."""
    if not valid_history:
        return []
    if isinstance(valid_history[0], dict):
        return [entry.get(metric, 0.0) for entry in valid_history]
    return valid_history



def plot_thesis_dashboard(results, output_path=None, title_suffix=""):
    """Create a compact dashboard with thesis-critical comparisons."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)
    if not strategies:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Relevance overview
    relevance_metrics = _available_metrics(
        stats_data, DEFAULT_RELEVANCE_METRICS, section="metrics"
    )
    if not relevance_metrics:
        return None
    primary_metric = _preferred_metric(stats_data, metric_base="ndcg")
    primary_quality_metric = _preferred_metric(
        stats_data, metric_base="item_coverage", section="quality_metrics"
    )
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
    primary_k = primary_metric.rsplit("@", 1)[-1]
    axes[0, 0].set_title(f"Relevance Metrics (@{primary_k}){title_suffix}")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(strategies, rotation=45, ha="right")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, axis="y", alpha=0.2)

    total_times = [_get_metric_value(stats_data[s], "total_time") for s in strategies]
    ndcg_values = [_get_metric_value(stats_data[s], primary_metric) for s in strategies]
    axes[0, 1].scatter(total_times, ndcg_values, color="steelblue", edgecolor="black")
    _label_points(axes[0, 1], strategies, total_times, ndcg_values)
    axes[0, 1].set_title(f"Efficiency Frontier{title_suffix}")
    axes[0, 1].set_xlabel("Total Time (s)")
    axes[0, 1].set_ylabel(primary_metric.upper())
    axes[0, 1].grid(True, alpha=0.3)

    coverage_values = [
        stats_data[s].get("quality_metrics", {}).get(primary_quality_metric, {}).get("mean", 0)
        for s in strategies
    ]
    axes[1, 0].scatter(
        coverage_values, ndcg_values, color="darkorange", edgecolor="black"
    )
    _label_points(axes[1, 0], strategies, coverage_values, ndcg_values)
    axes[1, 0].set_title(f"Relevance vs Coverage{title_suffix}")
    axes[1, 0].set_xlabel(primary_quality_metric)
    axes[1, 0].set_ylabel(primary_metric.upper())
    axes[1, 0].grid(True, alpha=0.3)

    # Sampling vs training cost
    sampling_times = [_get_metric_value(stats_data[s], "sampling_time") for s in strategies]
    training_times = [_get_metric_value(stats_data[s], "training_time") for s in strategies]
    axes[1, 1].barh(strategies, sampling_times, color="coral", label="Sampling")
    axes[1, 1].barh(
        strategies,
        training_times,
        left=sampling_times,
        color="steelblue",
        label="Training",
    )
    axes[1, 1].set_title(f"Time Breakdown{title_suffix}")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    _finalize_figure(fig, output_path)


def plot_training_dynamics(
    results,
    target_metric="ndcg@10",
    threshold_percentile=0.9,
    output_path=None,
    title_suffix="",
):
    """Plot validation dynamics, train loss curves, and convergence speed."""
    if "raw_results" not in results:
        return None

    raw_results = results["raw_results"]
    strategies = list(raw_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(strategies)))

    convergence = {}

    for idx, (strategy, runs) in enumerate(raw_results.items()):
        # --- Validation metric curves ---
        valid_series = [
            _extract_valid_series(run.get("valid_metrics", []), metric=target_metric)
            for run in runs
            if run.get("valid_metrics")
        ]
        valid_series = [series for series in valid_series if series]
        if valid_series:
            min_len = min(len(series) for series in valid_series)
            valid_array = np.array([series[:min_len] for series in valid_series])
            mean_valid = np.mean(valid_array, axis=0)
            epochs = np.arange(1, len(mean_valid) + 1)

            axes[0].plot(
                epochs,
                mean_valid,
                label=strategy,
                color=colors[idx],
                linewidth=2,
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

        # --- Train loss curves ---
        loss_series = [
            run.get("train_losses", [])
            for run in runs
            if run.get("train_losses")
        ]
        loss_series = [series for series in loss_series if series]
        if loss_series:
            min_len = min(len(series) for series in loss_series)
            loss_array = np.array([series[:min_len] for series in loss_series])
            mean_loss = np.mean(loss_array, axis=0)
            epochs = np.arange(1, len(mean_loss) + 1)

            axes[1].plot(
                epochs,
                mean_loss,
                label=strategy,
                color=colors[idx],
                linewidth=2,
            )

    axes[0].set_title(f"Validation {target_metric.upper()}{title_suffix}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(target_metric)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title(f"Train Loss{title_suffix}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    if convergence:
        conv_strategies = list(convergence.keys())
        means = [convergence[s]["mean"] for s in conv_strategies]
        stds = [convergence[s]["std"] for s in conv_strategies]
        bars = axes[2].bar(
            conv_strategies,
            means,
            yerr=stds,
            capsize=4,
            color="teal",
            edgecolor="black",
        )
        _annotate_vertical_bars(axes[2], bars, means, fmt="{:.1f}", offset=0.1)
        axes[2].set_title(
            f"Epochs to {threshold_percentile * 100:.0f}% of Best{title_suffix}"
        )
        axes[2].set_ylabel("Epoch")
        axes[2].tick_params(axis="x", rotation=45)
        axes[2].grid(True, axis="y", alpha=0.2)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return convergence


