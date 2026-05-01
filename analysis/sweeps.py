"""Parameter sweep and feature-uplift analysis."""

import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from .common import (
    DEFAULT_SWEEP_METRICS,
    FEATURE_UPLIFT_METRICS,
    _finalize_figure,
    _get_metric_value,
    _load_metadata_for_results,
    _sort_param_value,
    load_results,
)

def _sweep_ignored_keys():
    return {"seed", "device", "metrics", "topk", "valid_metric"}



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



def _sweep_x_axis(rows):
    param_values = sorted({row["param"] for row in rows}, key=_sort_param_value)
    return {
        "positions": np.arange(len(param_values)),
        "labels": [str(value) for value in param_values],
        "value_to_pos": {value: idx for idx, value in enumerate(param_values)},
    }


def _sweep_split_flags(rows, param):
    feature_states = sorted({row["feature_aware"] for row in rows})
    return {
        "feature": param != "feature_aware" and len(feature_states) > 1,
        "context": any(row["context"] for row in rows),
    }


def _group_sweep_rows(rows, split_flags):
    if not split_flags["feature"] and not split_flags["context"]:
        return {(None, ""): rows}

    grouped_rows = {}
    for row in rows:
        group_key = (
            row["feature_aware"] if split_flags["feature"] else None,
            row["context"] if split_flags["context"] else "",
        )
        grouped_rows.setdefault(group_key, []).append(row)
    return grouped_rows


def _draw_sweep_lines(ax, rows, strategies, value_to_pos, param, include_strategy=True):
    split_flags = _sweep_split_flags(rows, param)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, len(strategies)))
    legend_needed = False

    for color, strategy in zip(colors, strategies):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        if not strategy_rows:
            continue

        for group_key, group_rows in _group_sweep_rows(strategy_rows, split_flags).items():
            group_rows.sort(key=lambda row: _sort_param_value(row["param"]))
            xs = [value_to_pos[row["param"]] for row in group_rows]
            ys = [row["metric_value"] for row in group_rows]
            label = _line_label(
                strategy=strategy if include_strategy else None,
                feature_aware=group_key[0] if split_flags["feature"] else None,
                context=group_key[1] if split_flags["context"] else "",
            )
            line_style = "--" if group_key[0] else "-"
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=color,
                linestyle=line_style,
                label=label if label != "series" else None,
            )
            legend_needed = legend_needed or label != "series"
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

    return legend_needed


def _format_sweep_axis(ax, x_axis, param, ylabel, title):
    ax.set_xticks(x_axis["positions"])
    ax.set_xticklabels(x_axis["labels"], rotation=45, ha="right")
    ax.set_xlabel(str(param) if param is not None else "parameter")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _write_rows(csv_path, rows):
    if not csv_path or not rows:
        return
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    x_axis = _sweep_x_axis(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    if _draw_sweep_lines(
        ax, rows, [strategy], x_axis["value_to_pos"], param, include_strategy=False
    ):
        ax.legend(loc="best")

    param_label = str(param) if param is not None else "parameter"
    _format_sweep_axis(ax, x_axis, param, metric, f"{strategy}: {metric} across {param_label}")
    plt.tight_layout()
    _finalize_figure(fig, output_path)
    _write_rows(csv_path, rows)
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

    x_axis = _sweep_x_axis(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    if _draw_sweep_lines(ax, rows, strategies, x_axis["value_to_pos"], param):
        ax.legend(loc="best")

    param_label = str(param) if param is not None else "parameter"
    _format_sweep_axis(
        ax,
        x_axis,
        param,
        metric,
        f"Multi-strategy sweep: {metric} across {param_label}",
    )
    plt.tight_layout()
    _finalize_figure(fig, output_path)
    _write_rows(csv_path, rows)
    return rows


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
    rows_by_metric = {}
    for metric in metrics:
        rows, _, _ = _collect_sweep_rows_from_bundles(
            bundles, strategies, metric=metric, param=param
        )
        if rows:
            rows_by_metric[metric] = rows

    if not rows_by_metric:
        return None

    all_rows = [row for metric_rows in rows_by_metric.values() for row in metric_rows]
    x_axis = _sweep_x_axis(all_rows)
    metric_names = list(rows_by_metric.keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    csv_rows = []
    for ax, metric in zip(axes, metric_names):
        rows = rows_by_metric[metric]
        _draw_sweep_lines(ax, rows, strategies, x_axis["value_to_pos"], param)
        _format_sweep_axis(ax, x_axis, param, metric, metric)
        csv_rows.extend(rows)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best")
    fig.suptitle("Core Metric Sweep Overview")
    plt.tight_layout()
    _finalize_figure(fig, output_path)
    _write_rows(csv_path, csv_rows)
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



