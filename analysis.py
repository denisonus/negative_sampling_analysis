"""Analysis and Visualization for experimental results."""

import argparse
import csv
import os
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DEFAULT_RELEVANCE_METRICS = [
    "ndcg@10",
    "recall@10",
    "recall@20",
    "mrr@10",
    "hit@10",
]
DEFAULT_METRIC_BY_K_BASES = ["ndcg", "recall", "mrr"]
DEFAULT_SWEEP_METRICS = ["ndcg@10", "recall@10", "recall@20", "mrr@10"]
DEFAULT_BUCKET_METRICS = ["ndcg@10", "recall@10", "hit@10", "mrr@10"]
DEFAULT_QUALITY_METRICS = [
    "item_coverage@10",
    "novelty@10",
    "tail_percentage@10",
    "personalization@10",
]
SUMMARY_OPTIONAL_RELEVANCE_METRICS = ["precision@10", "map@10"]
FEATURE_UPLIFT_METRICS = ["ndcg@10", "recall@10", "recall@20", "mrr@10"]


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


def _available_metrics(stats_data, candidates, section="metrics"):
    """Keep only metrics that appear in at least one strategy."""
    return [
        metric
        for metric in candidates
        if any(metric in strategy_stats.get(section, {}) for strategy_stats in stats_data.values())
    ]


def _available_metric_ks(stats_data, metric_base, ks=(5, 10, 20)):
    """Return Ks that exist for the requested metric family."""
    return [
        k
        for k in ks
        if any(
            f"{metric_base}@{k}" in strategy_stats.get("metrics", {})
            for strategy_stats in stats_data.values()
        )
    ]


def _bucket_sort_key(bucket_label):
    order = {"0": 0, "1-5": 1, "6-20": 2, "21+": 3}
    return order.get(bucket_label, len(order))


def _collect_bucket_labels(stats_data):
    labels = set()
    for strategy_stats in stats_data.values():
        labels.update(strategy_stats.get("bucket_metrics", {}).keys())
    return sorted(labels, key=_bucket_sort_key)


def _available_bucket_metrics(stats_data, candidates):
    """Keep only bucket metrics that appear in at least one bucket."""
    return [
        metric
        for metric in candidates
        if any(
            metric in bucket_stats
            for strategy_stats in stats_data.values()
            for bucket_stats in strategy_stats.get("bucket_metrics", {}).values()
        )
    ]


def _feature_aware_from_metadata(metadata):
    """Extract feature-aware flag from sibling metadata when available."""
    if not metadata:
        return None
    config = metadata.get("config", {})
    if "feature_aware" in config:
        return bool(config["feature_aware"])
    dataset_stats = metadata.get("dataset_stats", {})
    if "feature_aware" in dataset_stats:
        return bool(dataset_stats["feature_aware"])
    return None


def _title_suffix_from_metadata(metadata):
    """Build a concise title suffix for per-run reports."""
    feature_aware = _feature_aware_from_metadata(metadata)
    if feature_aware is None:
        return ""
    return " [feature-aware]" if feature_aware else " [id-only]"


def _sweep_ignored_keys():
    return {"seed", "device", "metrics", "topk", "valid_metric"}


def _sort_param_value(value):
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    if value is None:
        return (1, "")
    return (2, str(value))


def _infer_context_keys(bundles, primary_param=None):
    """Infer additional varying config keys beyond the main sweep parameter."""
    if not bundles:
        return []

    ignored_keys = _sweep_ignored_keys() | {"feature_aware"}
    if primary_param is not None:
        ignored_keys.add(primary_param)

    context_keys = []
    all_keys = set().union(*(bundle["config"].keys() for bundle in bundles))
    for key in sorted(all_keys):
        if key in ignored_keys:
            continue
        values = {
            json.dumps(bundle["config"].get(key, None), sort_keys=True)
            for bundle in bundles
        }
        if len(values) > 1:
            context_keys.append(key)
    return context_keys


def _format_context(config, context_keys):
    """Format extra varying config fields into a compact label."""
    if not context_keys:
        return ""
    return ", ".join(f"{key}={config.get(key)}" for key in context_keys)


def _line_label(strategy=None, feature_aware=None, context=""):
    """Build a readable legend label for sweep lines."""
    parts = []
    if strategy is not None:
        parts.append(strategy)
    if feature_aware is not None:
        parts.append("feat" if feature_aware else "id")
    if context:
        parts.append(context)
    return " | ".join(parts) if parts else "series"


def _load_sweep_bundles(results_files):
    """Load results files together with sibling metadata/config."""
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
    return bundles


def _infer_sweep_param(bundles, param=None):
    """Infer the single varying config parameter across sweep bundles."""
    if not bundles:
        return param
    if param is not None:
        return param

    ignored_keys = _sweep_ignored_keys()
    candidate_params = []
    all_keys = set().union(*(bundle["config"].keys() for bundle in bundles))
    for key in sorted(all_keys):
        if key in ignored_keys:
            continue
        values = {
            json.dumps(bundle["config"].get(key, None), sort_keys=True)
            for bundle in bundles
        }
        if len(values) > 1:
            candidate_params.append(key)
    if len(candidate_params) != 1:
        raise ValueError(
            "Could not infer a single varying parameter. Pass --sweep_param explicitly."
        )
    return candidate_params[0]


def _collect_sweep_rows_from_bundles(bundles, strategies, metric="ndcg@10", param=None):
    """Build simple per-bundle sweep rows for the requested strategies and metric."""
    if not bundles:
        return [], param, []

    param = _infer_sweep_param(bundles, param=param)
    context_keys = _infer_context_keys(bundles, primary_param=param)
    rows = []
    for bundle in bundles:
        stats = bundle["results"].get("statistics", {})
        param_value = bundle["config"].get(param)
        feature_aware = bool(bundle["config"].get("feature_aware", False))
        context = _format_context(bundle["config"], context_keys)
        for strategy in strategies:
            if strategy not in stats:
                continue
            rows.append(
                {
                    "param": param_value,
                    "strategy": strategy,
                    "feature_aware": feature_aware,
                    "context": context,
                    "metric_name": metric,
                    "metric_value": _get_metric_value(stats[strategy], metric),
                    "results_file": bundle["results_file"],
                }
            )

    return rows, param, context_keys


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
    results,
    metrics=None,
    output_path=None,
    title_suffix="",
):
    """Create grouped bar chart comparing all metrics with error bars."""
    stats_data = results["statistics"]
    if metrics is None:
        metrics = list(DEFAULT_RELEVANCE_METRICS)
    metrics = _available_metrics(stats_data, metrics, section="metrics")
    if not metrics:
        return None

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
    ax.set_title(f"Comparison of Negative Sampling Strategies{title_suffix}")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return metrics


def plot_metric_by_k(
    results,
    metric_bases=None,
    ks=(5, 10, 20),
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


def plot_quality_metrics(
    results,
    metrics=None,
    output_path=None,
    title_suffix="",
):
    """Create grouped bar chart for recommendation-quality metrics."""
    stats_data = results["statistics"]
    if metrics is None:
        metrics = list(DEFAULT_QUALITY_METRICS)
    metrics = _available_metrics(stats_data, metrics, section="quality_metrics")
    if not metrics:
        return None

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
    ax.set_title(f"Recommendation Quality Metrics{title_suffix}")
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
    title_suffix="",
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

    fig.suptitle(f"Recommendation Quality Metrics{title_suffix}")
    plt.tight_layout()
    _finalize_figure(fig, output_path)


def plot_competitive_quality(
    results,
    primary_metric="ndcg@10",
    min_relative=0.97,
    metrics=None,
    output_path=None,
    title_suffix="",
):
    """Plot quality metrics only for strategies that remain close to the best relevance."""
    if metrics is None:
        metrics = ["item_coverage@10", "novelty@10", "personalization@10"]

    stats_data = results["statistics"]
    if not stats_data:
        return []

    best_primary = max(_get_metric_value(stats, primary_metric) for stats in stats_data.values())
    threshold = best_primary * min_relative
    strategies = [
        strategy
        for strategy in _sorted_strategies(stats_data, metric=primary_metric)
        if _get_metric_value(stats_data[strategy], primary_metric) >= threshold
    ]

    if not strategies:
        return []

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = plt.colormaps["Set2"](np.linspace(0, 1, len(metrics)))
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

    fig.suptitle(
        f"Competitive Quality ({primary_metric} >= {min_relative:.0%} of best){title_suffix}"
    )
    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return strategies


def plot_metric_tradeoff(
    results,
    x_metric,
    y_metric="ndcg@10",
    output_path=None,
    title=None,
    title_suffix="",
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
    ax.set_title((title or f"Tradeoff: {y_metric} vs {x_metric}") + title_suffix)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _finalize_figure(fig, output_path)


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


def save_user_bucket_metrics_table(results, output_path):
    """Persist aggregated per-user activity bucket metrics to CSV."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)
    bucket_labels = _collect_bucket_labels(stats_data)
    metrics = _available_bucket_metrics(stats_data, DEFAULT_BUCKET_METRICS)

    rows = []
    for strategy in strategies:
        strategy_bucket_metrics = stats_data[strategy].get("bucket_metrics", {})
        for bucket_label in bucket_labels:
            bucket_metrics = strategy_bucket_metrics.get(bucket_label, {})
            for metric in metrics:
                metric_stats = bucket_metrics.get(metric)
                if metric_stats is None:
                    continue
                rows.append(
                    {
                        "strategy": strategy,
                        "bucket": bucket_label,
                        "metric": metric,
                        "mean": metric_stats.get("mean", 0.0),
                        "std": metric_stats.get("std", 0.0),
                        "ci_lower": metric_stats.get("ci_lower", 0.0),
                        "ci_upper": metric_stats.get("ci_upper", 0.0),
                        "num_runs": len(metric_stats.get("values", [])),
                    }
                )

    if not rows:
        return []

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def plot_user_bucket_metrics(results, output_path=None, title_suffix=""):
    """Plot grouped bucket metrics for the main @10 user-activity metrics."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)
    bucket_labels = _collect_bucket_labels(stats_data)
    if not bucket_labels:
        return None

    metrics = _available_bucket_metrics(stats_data, DEFAULT_BUCKET_METRICS)
    if not metrics:
        return None
    colors = plt.colormaps["Set2"](np.linspace(0, 1, len(bucket_labels)))
    if len(metrics) <= 3:
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), sharex=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(len(strategies))
    width = 0.8 / max(len(bucket_labels), 1)

    for ax, metric in zip(axes, metrics):
        for idx, bucket_label in enumerate(bucket_labels):
            values = [
                stats_data[strategy]
                .get("bucket_metrics", {})
                .get(bucket_label, {})
                .get(metric, {})
                .get("mean", 0.0)
                for strategy in strategies
            ]
            errors = [
                stats_data[strategy]
                .get("bucket_metrics", {})
                .get(bucket_label, {})
                .get(metric, {})
                .get("std", 0.0)
                for strategy in strategies
            ]
            offset = (idx - len(bucket_labels) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                values,
                width,
                label=bucket_label,
                color=colors[idx],
                yerr=errors,
                capsize=3,
            )

        ax.set_title(metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    axes[0].set_ylabel("Metric Value")
    fig.suptitle(f"Per-User Activity Bucket Metrics{title_suffix}")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Train interactions", loc="upper center", ncol=len(bucket_labels))
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    _finalize_figure(fig, output_path)
    return bucket_labels


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
            epochs = np.arange(len(mean_loss))

            axes[0].plot(
                epochs, mean_loss, label=strategy, color=colors[idx], linewidth=2
            )

        if all_valid and all_valid[0]:
            valid_series = [_extract_valid_series(v) for v in all_valid if v]
            min_len = min(len(v) for v in valid_series if v)
            valid_array = np.array([v[:min_len] for v in valid_series if v])

            mean_valid = np.mean(valid_array, axis=0)
            epochs = np.arange(len(mean_valid))

            axes[1].plot(
                epochs, mean_valid, label=strategy, color=colors[idx], linewidth=2
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
    axes[0, 0].set_title(f"Relevance Metrics (@10){title_suffix}")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(strategies, rotation=45, ha="right")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, axis="y", alpha=0.2)

    total_times = [
        stats_data[s]["timing"]["total_time"]["mean"] for s in strategies
    ]
    ndcg_values = [_get_metric_value(stats_data[s], "ndcg@10") for s in strategies]
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
    axes[0, 1].set_title(f"Efficiency Frontier{title_suffix}")
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
    axes[1, 0].set_title(f"Relevance vs Coverage{title_suffix}")
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

    axes[0].set_title(f"Validation {target_metric.upper()} Dynamics{title_suffix}")
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
            f"Epochs to {threshold_percentile * 100:.0f}% of Best {target_metric.upper()}{title_suffix}"
        )
        axes[1].set_ylabel("Epoch")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, axis="y", alpha=0.2)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return convergence


def save_summary_table(results, output_path, metadata=None):
    """Persist a compact strategy comparison table."""
    stats_data = results["statistics"]
    strategies = _sorted_strategies(stats_data)
    rows = []
    feature_aware = _feature_aware_from_metadata(metadata)
    relevance_metrics = _available_metrics(
        stats_data,
        [
            "ndcg@10",
            "recall@10",
            "recall@20",
            "mrr@10",
            "hit@10",
            *SUMMARY_OPTIONAL_RELEVANCE_METRICS,
        ],
        section="metrics",
    )
    quality_metrics = _available_metrics(
        stats_data,
        ["item_coverage@10", "novelty@10", "tail_percentage@10", "personalization@10"],
        section="quality_metrics",
    )

    for strategy in strategies:
        strategy_stats = stats_data[strategy]
        total_time = strategy_stats["timing"]["total_time"]["mean"]
        sampling_time = strategy_stats["timing"]["sampling_time"]["mean"]
        row = {"strategy": strategy}
        for metric in relevance_metrics:
            row[metric] = _get_metric_value(strategy_stats, metric)
        for metric in quality_metrics:
            row[metric] = _get_metric_value(strategy_stats, metric)
        row["total_time"] = total_time
        row["sampling_time"] = sampling_time
        row["training_time"] = strategy_stats["timing"]["training_time"]["mean"]
        row["sampling_share"] = (sampling_time / total_time) if total_time > 0 else 0.0
        row["feature_aware"] = feature_aware
        rows.append(row)

    if not rows:
        return

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_competitive_summary(
    results,
    output_path,
    primary_metric="ndcg@10",
    min_relative=0.97,
    metadata=None,
):
    """Persist a compact table for strategies that remain close to the best primary metric."""
    stats_data = results["statistics"]
    if not stats_data:
        return []

    best_primary = max(_get_metric_value(stats, primary_metric) for stats in stats_data.values())
    threshold = best_primary * min_relative
    strategies = [
        strategy
        for strategy in _sorted_strategies(stats_data, metric=primary_metric)
        if _get_metric_value(stats_data[strategy], primary_metric) >= threshold
    ]

    rows = []
    feature_aware = _feature_aware_from_metadata(metadata)
    relevance_metrics = [
        metric
        for metric in ["recall@10", "recall@20", "mrr@10", "hit@10"]
        if _available_metrics(stats_data, [metric], section="metrics")
    ]
    for strategy in strategies:
        strategy_stats = stats_data[strategy]
        row = {
            "strategy": strategy,
            primary_metric: _get_metric_value(strategy_stats, primary_metric),
        }
        for metric in relevance_metrics:
            row[metric] = _get_metric_value(strategy_stats, metric)
        row["item_coverage@10"] = _get_metric_value(strategy_stats, "item_coverage@10")
        row["novelty@10"] = _get_metric_value(strategy_stats, "novelty@10")
        row["personalization@10"] = _get_metric_value(strategy_stats, "personalization@10")
        row["total_time"] = _get_metric_value(strategy_stats, "total_time")
        row["feature_aware"] = feature_aware
        rows.append(row)

    if not rows:
        return []

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_significance_table(
    results, output_path, metric="ndcg@10", baseline="uniform", quiet=True
):
    """Persist paired significance results when multi-run data is available."""
    table = statistical_significance_test(
        results, metric=metric, baseline=baseline, quiet=quiet
    )
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
    bundles = _load_sweep_bundles(results_files)
    rows, param, _ = _collect_sweep_rows_from_bundles(
        bundles, [strategy], metric=metric, param=param
    )

    if not rows:
        raise ValueError(f"Strategy '{strategy}' not found in provided result files.")

    rows.sort(key=lambda row: (_sort_param_value(row["param"]), row["context"]))

    param_label = str(param) if param is not None else "parameter"

    param_values = sorted({row["param"] for row in rows}, key=_sort_param_value)
    x_positions = np.arange(len(param_values))
    x_labels = [str(value) for value in param_values]
    value_to_pos = {value: idx for idx, value in enumerate(param_values)}
    feature_states = sorted({row["feature_aware"] for row in rows})
    has_feature_split = param != "feature_aware" and len(feature_states) > 1
    has_context_split = any(row["context"] for row in rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    if has_feature_split or has_context_split:
        grouped_rows = {}
        for row in rows:
            group_key = (
                row["feature_aware"] if has_feature_split else None,
                row["context"] if has_context_split else "",
            )
            grouped_rows.setdefault(group_key, []).append(row)

        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(grouped_rows)))
        for color, (group_key, group_rows) in zip(colors, grouped_rows.items()):
            group_rows.sort(key=lambda row: _sort_param_value(row["param"]))
            xs = [value_to_pos[row["param"]] for row in group_rows]
            ys = [row["metric_value"] for row in group_rows]
            label = _line_label(
                feature_aware=group_key[0] if has_feature_split else None,
                context=group_key[1] if has_context_split else "",
            )
            ax.plot(xs, ys, marker="o", color=color, linewidth=2, label=label)
            for x_position, y_value in zip(xs, ys):
                ax.text(
                    x_position,
                    y_value,
                    f"{y_value:.3f}",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color=color,
                )
        ax.legend(loc="best")
    else:
        y_values = [row["metric_value"] for row in rows]
        row_positions = [value_to_pos[row["param"]] for row in rows]
        ax.plot(row_positions, y_values, marker="o", color="steelblue", linewidth=2)
        for x_position, row in zip(row_positions, rows):
            ax.text(
                x_position,
                row["metric_value"],
                f"{row['metric_value']:.3f}",
                fontsize=8,
                ha="center",
                va="bottom",
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(param_label)
    ax.set_ylabel(metric)
    ax.set_title(f"{strategy}: {metric} across {param_label}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _finalize_figure(fig, output_path)

    if csv_path:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return rows


def plot_multi_strategy_sweep(
    results_files,
    strategies,
    metric="ndcg@10",
    param=None,
    output_path=None,
    csv_path=None,
):
    """Overlay multiple strategies on the same parameter sweep."""
    bundles = _load_sweep_bundles(results_files)
    rows, param, _ = _collect_sweep_rows_from_bundles(
        bundles, strategies, metric=metric, param=param
    )

    if not rows:
        raise ValueError("None of the requested strategies were found in the provided results.")

    param_label = str(param) if param is not None else "parameter"

    param_values = sorted({row["param"] for row in rows}, key=_sort_param_value)
    x_positions = np.arange(len(param_values))
    x_labels = [str(value) for value in param_values]
    value_to_pos = {value: idx for idx, value in enumerate(param_values)}
    feature_states = sorted({row["feature_aware"] for row in rows})
    has_feature_split = param != "feature_aware" and len(feature_states) > 1
    has_context_split = any(row["context"] for row in rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(strategies)))
    csv_rows = []

    for color, strategy in zip(colors, strategies):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        if not strategy_rows:
            continue
        grouped_rows = {}
        if has_feature_split or has_context_split:
            for row in strategy_rows:
                group_key = (
                    row["feature_aware"] if has_feature_split else None,
                    row["context"] if has_context_split else "",
                )
                grouped_rows.setdefault(group_key, []).append(row)
        else:
            grouped_rows[(None, "")] = strategy_rows

        for group_key, group_rows in grouped_rows.items():
            group_rows.sort(key=lambda row: _sort_param_value(row["param"]))
            xs = [value_to_pos[row["param"]] for row in group_rows]
            ys = [row["metric_value"] for row in group_rows]
            line_style = "--" if group_key[0] else "-"
            label = _line_label(
                strategy=strategy,
                feature_aware=group_key[0] if has_feature_split else None,
                context=group_key[1] if has_context_split else "",
            )
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=color,
                linestyle=line_style,
                label=label,
            )
            for x_value, y_value in zip(xs, ys):
                ax.text(
                    x_value,
                    y_value,
                    f"{y_value:.3f}",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color=color,
                )
        csv_rows.extend(strategy_rows)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(param_label)
    ax.set_ylabel(metric)
    ax.set_title(f"Multi-strategy sweep: {metric} across {param_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    _finalize_figure(fig, output_path)

    if csv_path:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    return csv_rows


def plot_multi_metric_sweep(
    results_files,
    strategies,
    metrics=None,
    param=None,
    output_path=None,
    csv_path=None,
):
    """Create a compact multi-panel sweep view for the main report metrics."""
    bundles = _load_sweep_bundles(results_files)
    if metrics is None:
        metrics = list(DEFAULT_SWEEP_METRICS)

    param = _infer_sweep_param(bundles, param=param)
    context_keys = _infer_context_keys(bundles, primary_param=param)
    rows_by_metric = {}
    for metric in metrics:
        rows, _, _ = _collect_sweep_rows_from_bundles(
            bundles, strategies, metric=metric, param=param
        )
        if rows:
            rows_by_metric[metric] = rows

    if not rows_by_metric:
        return None

    param_values = sorted(
        {
            row["param"]
            for metric_rows in rows_by_metric.values()
            for row in metric_rows
        },
        key=_sort_param_value,
    )
    x_positions = np.arange(len(param_values))
    x_labels = [str(value) for value in param_values]
    value_to_pos = {value: idx for idx, value in enumerate(param_values)}

    metric_names = list(rows_by_metric.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(strategies)))
    csv_rows = []

    for ax, metric in zip(axes, metric_names):
        rows = rows_by_metric[metric]
        feature_states = sorted({row["feature_aware"] for row in rows})
        has_feature_split = param != "feature_aware" and len(feature_states) > 1
        has_context_split = any(row["context"] for row in rows)

        for color, strategy in zip(colors, strategies):
            strategy_rows = [row for row in rows if row["strategy"] == strategy]
            if not strategy_rows:
                continue

            grouped_rows = {}
            if has_feature_split or has_context_split:
                for row in strategy_rows:
                    group_key = (
                        row["feature_aware"] if has_feature_split else None,
                        row["context"] if has_context_split else "",
                    )
                    grouped_rows.setdefault(group_key, []).append(row)
            else:
                grouped_rows[(None, "")] = strategy_rows

            for group_key, group_rows in grouped_rows.items():
                group_rows.sort(key=lambda row: _sort_param_value(row["param"]))
                xs = [value_to_pos[row["param"]] for row in group_rows]
                ys = [row["metric_value"] for row in group_rows]
                line_style = "--" if group_key[0] else "-"
                label = _line_label(
                    strategy=strategy,
                    feature_aware=group_key[0] if has_feature_split else None,
                    context=group_key[1] if has_context_split else "",
                )
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=2,
                    color=color,
                    linestyle=line_style,
                    label=label,
                )
                for x_value, y_value in zip(xs, ys):
                    ax.text(
                        x_value,
                        y_value,
                        f"{y_value:.3f}",
                        fontsize=8,
                        ha="center",
                        va="bottom",
                        color=color,
                    )

            csv_rows.extend(strategy_rows)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(str(param) if param is not None else "parameter")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best")
    fig.suptitle("Core Metric Sweep Overview")
    plt.tight_layout()
    _finalize_figure(fig, output_path)

    if csv_path:
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    return csv_rows


def build_feature_uplift_rows(results_files, strategies=None, param=None):
    """Build paired id-only vs feature-aware comparison rows."""
    bundles = _load_sweep_bundles(results_files)
    if not bundles:
        return None, None

    param = _infer_sweep_param(bundles, param=param)
    context_keys = _infer_context_keys(bundles, primary_param=param)

    base_rows = []
    for bundle in bundles:
        stats = bundle["results"].get("statistics", {})
        available_strategies = strategies or sorted(stats.keys())
        feature_aware = bool(bundle["config"].get("feature_aware", False))
        param_value = bundle["config"].get(param)
        context = _format_context(bundle["config"], context_keys)
        for strategy in available_strategies:
            if strategy not in stats:
                continue
            strategy_stats = stats[strategy]
            row = {
                "strategy": strategy,
                "sweep_param": param,
                "sweep_value": param_value,
                "context": context,
                "feature_aware": feature_aware,
                "results_file": bundle["results_file"],
            }
            for metric in FEATURE_UPLIFT_METRICS:
                if metric in strategy_stats.get("metrics", {}):
                    row[metric] = _get_metric_value(strategy_stats, metric)
            row["item_coverage@10"] = _get_metric_value(strategy_stats, "item_coverage@10")
            row["novelty@10"] = _get_metric_value(strategy_stats, "novelty@10")
            row["total_time"] = _get_metric_value(strategy_stats, "total_time")
            base_rows.append(row)

    if not base_rows:
        return None, param

    if len({row["feature_aware"] for row in base_rows}) < 2:
        return None, param

    grouped = {}
    for row in base_rows:
        pair_key = (
            row["strategy"],
            row["context"],
            None if param == "feature_aware" else row["sweep_value"],
        )
        grouped.setdefault(pair_key, {})[row["feature_aware"]] = row

    uplift_rows = []
    for (strategy, context, sweep_value), pair in grouped.items():
        if False not in pair or True not in pair:
            continue
        id_row = pair[False]
        feature_row = pair[True]
        uplift_rows.append(
            {
                "strategy": strategy,
                "sweep_param": param,
                "sweep_value": sweep_value,
                "context": context,
                "id_item_coverage@10": id_row["item_coverage@10"],
                "feature_item_coverage@10": feature_row["item_coverage@10"],
                "delta_item_coverage@10": feature_row["item_coverage@10"]
                - id_row["item_coverage@10"],
                "id_novelty@10": id_row["novelty@10"],
                "feature_novelty@10": feature_row["novelty@10"],
                "delta_novelty@10": feature_row["novelty@10"] - id_row["novelty@10"],
                "id_total_time": id_row["total_time"],
                "feature_total_time": feature_row["total_time"],
                "delta_total_time": feature_row["total_time"] - id_row["total_time"],
            }
        )
        for metric in FEATURE_UPLIFT_METRICS:
            if metric not in id_row or metric not in feature_row:
                continue
            uplift_rows[-1][f"id_{metric}"] = id_row[metric]
            uplift_rows[-1][f"feature_{metric}"] = feature_row[metric]
            uplift_rows[-1][f"delta_{metric}"] = feature_row[metric] - id_row[metric]

    if not uplift_rows:
        return None, param

    uplift_rows.sort(
        key=lambda row: (
            row["strategy"],
            _sort_param_value(row["sweep_value"]),
            row["context"],
        )
    )
    return uplift_rows, param


def save_feature_uplift_table(results_files, output_path, strategies=None, param=None):
    """Persist paired feature-aware uplift statistics across runs."""
    rows, _ = build_feature_uplift_rows(results_files, strategies=strategies, param=param)
    if not rows:
        return None

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _feature_uplift_label(row):
    """Build a compact x-axis label for feature uplift plots."""
    label = row["strategy"]
    if row.get("sweep_value") is not None:
        label += f"\n{row['sweep_param']}={row['sweep_value']}"
    if row.get("context"):
        label += f"\n{row['context']}"
    return label


def plot_feature_uplift(results_files, output_path, strategies=None, param=None):
    """Plot feature-aware uplift across the core relevance metrics."""
    rows, _ = build_feature_uplift_rows(results_files, strategies=strategies, param=param)
    if not rows:
        return None

    labels = [_feature_uplift_label(row) for row in rows]
    x = np.arange(len(rows))
    delta_metrics = [
        metric
        for metric in [
            "delta_ndcg@10",
            "delta_recall@10",
            "delta_recall@20",
            "delta_mrr@10",
        ]
        if any(metric in row for row in rows)
    ]
    width = 0.8 / max(len(delta_metrics), 1)

    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 1.4), 6))
    colors = plt.colormaps["Set2"](np.linspace(0, 1, len(delta_metrics)))
    for idx, metric in enumerate(delta_metrics):
        values = [row[metric] for row in rows]
        offset = (idx - len(delta_metrics) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            values,
            width,
            label=metric,
            color=colors[idx],
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Feature-aware delta")
    ax.set_title("Feature-aware uplift by strategy")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return rows


def plot_feature_quality_tradeoff(
    results_files,
    output_path,
    strategies=None,
    param=None,
    quality_metric="delta_item_coverage@10",
):
    """Plot whether feature-aware gains also change recommendation breadth."""
    rows, _ = build_feature_uplift_rows(results_files, strategies=strategies, param=param)
    if not rows:
        return None

    x_values = [row[quality_metric] for row in rows]
    y_values = [row["delta_ndcg@10"] for row in rows]
    sizes = [
        max(60, abs(row["delta_total_time"]) * 20 + 60)
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x_values, y_values, s=sizes, color="seagreen", edgecolor="black")
    for x_value, y_value, row in zip(x_values, y_values, rows):
        ax.text(
            x_value,
            y_value,
            _feature_uplift_label(row).replace("\n", " | "),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel(quality_metric)
    ax.set_ylabel("delta_ndcg@10")
    ax.set_title("Feature-aware relevance vs quality tradeoff")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _finalize_figure(fig, output_path)
    return rows


def statistical_significance_test(results, metric="ndcg@10", baseline="uniform", quiet=False):
    """Perform paired t-tests comparing each strategy to baseline."""
    if "statistics" not in results:
        if not quiet:
            print("Statistical tests require multi-run experiment data")
        return None

    stats_data = results["statistics"]

    if baseline not in stats_data:
        if not quiet:
            print(f"Baseline strategy '{baseline}' not found")
        return None

    baseline_values = stats_data[baseline]["metrics"].get(metric, {}).get("values", [])
    if len(baseline_values) < 2:
        if not quiet:
            print(f"Baseline strategy '{baseline}' has insufficient runs for significance testing")
        return None

    if not quiet:
        print(f"\nStatistical Significance Tests vs {baseline} on {metric}")
        print("=" * 60)

    results_table = []
    skipped = []
    for strategy in stats_data:
        if strategy == baseline:
            continue

        strategy_values = (
            stats_data[strategy]["metrics"].get(metric, {}).get("values", [])
        )

        if len(baseline_values) != len(strategy_values) or len(baseline_values) < 2:
            skipped.append(strategy)
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

        if not quiet:
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

    if not results_table:
        if not quiet:
            print("No valid paired comparisons available")
        return None

    if not quiet:
        if skipped:
            print("\nSkipped:")
            for strategy in skipped:
                print(f"{strategy}: mismatched or insufficient runs")
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
    metadata = _load_metadata_for_results(results_file)
    title_suffix = _title_suffix_from_metadata(metadata)
    save_summary_table(
        results, os.path.join(output_dir, "summary_metrics.csv"), metadata=metadata
    )
    save_competitive_summary(
        results,
        os.path.join(output_dir, "competitive_summary.csv"),
        metadata=metadata,
    )
    plot_thesis_dashboard(
        results,
        output_path=os.path.join(output_dir, "dashboard.png"),
        title_suffix=title_suffix,
    )
    plot_all_metrics(
        results,
        output_path=os.path.join(output_dir, "relevance_metrics.png"),
        title_suffix=title_suffix,
    )
    plot_metric_by_k(
        results,
        output_path=os.path.join(output_dir, "metric_by_k.png"),
        title_suffix=title_suffix,
    )
    has_quality_metrics = any(
        stats_data.get("quality_metrics") for stats_data in results["statistics"].values()
    )
    if has_quality_metrics:
        plot_quality_small_multiples(
            results,
            output_path=os.path.join(output_dir, "quality_metrics.png"),
            title_suffix=title_suffix,
        )
        plot_competitive_quality(
            results,
            output_path=os.path.join(output_dir, "competitive_quality.png"),
            title_suffix=title_suffix,
        )
        plot_metric_tradeoff(
            results,
            x_metric="item_coverage@10",
            y_metric="ndcg@10",
            output_path=os.path.join(output_dir, "ndcg_vs_coverage.png"),
            title="NDCG@10 vs Item Coverage@10",
            title_suffix=title_suffix,
        )
        plot_metric_tradeoff(
            results,
            x_metric="novelty@10",
            y_metric="ndcg@10",
            output_path=os.path.join(output_dir, "ndcg_vs_novelty.png"),
            title="NDCG@10 vs Novelty@10",
            title_suffix=title_suffix,
        )
    has_bucket_metrics = any(
        stats_data.get("bucket_metrics") for stats_data in results["statistics"].values()
    )
    if has_bucket_metrics:
        save_user_bucket_metrics_table(
            results, os.path.join(output_dir, "user_bucket_metrics.csv")
        )
        plot_user_bucket_metrics(
            results,
            output_path=os.path.join(output_dir, "user_bucket_metrics.png"),
            title_suffix=title_suffix,
        )
        plot_user_bucket_delta_heatmap(
            results,
            baseline="uniform",
            metric="ndcg@10",
            output_path=os.path.join(output_dir, "user_bucket_ndcg10_delta_heatmap.png"),
            title_suffix=title_suffix,
        )
    plot_metric_tradeoff(
        results,
        x_metric="total_time",
        y_metric="ndcg@10",
        output_path=os.path.join(output_dir, "ndcg_vs_time.png"),
        title="NDCG@10 vs Total Time",
        title_suffix=title_suffix,
    )
    plot_ablation_delta(
        results,
        baseline="uniform",
        metric="ndcg@10",
        output_path=os.path.join(output_dir, "ndcg10_delta_vs_uniform.png"),
        title_suffix=title_suffix,
    )

    if "raw_results" in results:
        plot_training_dynamics(
            results,
            output_path=os.path.join(output_dir, "training_dynamics.png"),
            title_suffix=title_suffix,
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
    parser.add_argument("--sweep_strategies", nargs="+", default=None)
    parser.add_argument("--sweep_metric", type=str, default="ndcg@10")
    parser.add_argument("--sweep_param", type=str, default=None)
    args = parser.parse_args()

    if len(args.results_files) == 1 and args.sweep_strategy is None:
        generate_full_report(args.results_files[0], args.output_dir)
    else:
        if args.sweep_strategy and args.sweep_strategies:
            raise SystemExit("Use either --sweep_strategy or --sweep_strategies, not both.")
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(args.results_files[0]), "sweep_analysis"
        )
        os.makedirs(output_dir, exist_ok=True)
        if args.sweep_strategy is None and args.sweep_strategies is None:
            raise SystemExit(
                "Multiple results files require --sweep_strategy or --sweep_strategies."
            )
        plt.switch_backend("Agg")
        if args.sweep_strategies:
            plot_multi_strategy_sweep(
                args.results_files,
                strategies=args.sweep_strategies,
                metric=args.sweep_metric,
                param=args.sweep_param,
                output_path=os.path.join(output_dir, "parameter_sweep_multi.png"),
                csv_path=os.path.join(output_dir, "parameter_sweep_multi.csv"),
            )
            plot_multi_metric_sweep(
                args.results_files,
                strategies=args.sweep_strategies,
                metrics=DEFAULT_SWEEP_METRICS,
                param=args.sweep_param,
                output_path=os.path.join(output_dir, "parameter_sweep_core_metrics.png"),
                csv_path=os.path.join(output_dir, "parameter_sweep_core_metrics.csv"),
            )
            save_feature_uplift_table(
                args.results_files,
                os.path.join(output_dir, "feature_uplift.csv"),
                strategies=args.sweep_strategies,
                param=args.sweep_param,
            )
            plot_feature_uplift(
                args.results_files,
                os.path.join(output_dir, "feature_uplift.png"),
                strategies=args.sweep_strategies,
                param=args.sweep_param,
            )
            plot_feature_quality_tradeoff(
                args.results_files,
                os.path.join(output_dir, "feature_quality_tradeoff.png"),
                strategies=args.sweep_strategies,
                param=args.sweep_param,
            )
        else:
            plot_parameter_sweep(
                args.results_files,
                strategy=args.sweep_strategy,
                metric=args.sweep_metric,
                param=args.sweep_param,
                output_path=os.path.join(output_dir, "parameter_sweep.png"),
                csv_path=os.path.join(output_dir, "parameter_sweep.csv"),
            )
            plot_multi_metric_sweep(
                args.results_files,
                strategies=[args.sweep_strategy],
                metrics=DEFAULT_SWEEP_METRICS,
                param=args.sweep_param,
                output_path=os.path.join(output_dir, "parameter_sweep_core_metrics.png"),
                csv_path=os.path.join(output_dir, "parameter_sweep_core_metrics.csv"),
            )
            save_feature_uplift_table(
                args.results_files,
                os.path.join(output_dir, "feature_uplift.csv"),
                strategies=[args.sweep_strategy],
                param=args.sweep_param,
            )
            plot_feature_uplift(
                args.results_files,
                os.path.join(output_dir, "feature_uplift.png"),
                strategies=[args.sweep_strategy],
                param=args.sweep_param,
            )
            plot_feature_quality_tradeoff(
                args.results_files,
                os.path.join(output_dir, "feature_quality_tradeoff.png"),
                strategies=[args.sweep_strategy],
                param=args.sweep_param,
            )
