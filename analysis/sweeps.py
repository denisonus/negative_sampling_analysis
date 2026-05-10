"""Parameter sweep analysis."""

import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from .common import (
    DEFAULT_SWEEP_METRICS,
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

    ignored_keys = _sweep_ignored_keys()
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


def _line_label(strategy=None, context=""):
    """Build a readable legend label for sweep lines."""
    parts = []
    if strategy is not None:
        parts.append(strategy)
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


def _has_metric_value(strategy_stats, metric):
    """Return whether a metric exists in any supported stats section."""
    return (
        metric in strategy_stats.get("metrics", {})
        or metric in strategy_stats.get("quality_metrics", {})
        or metric in strategy_stats.get("timing", {})
    )


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
        context = _format_context(bundle["config"], context_keys)
        for strategy in strategies:
            if strategy not in stats:
                continue
            if not _has_metric_value(stats[strategy], metric):
                continue
            rows.append(
                {
                    "param": param_value,
                    "strategy": strategy,
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
    return {
        "context": any(row["context"] for row in rows),
    }


def _group_sweep_rows(rows, split_flags):
    if not split_flags["context"]:
        return {(None, ""): rows}

    grouped_rows = {}
    for row in rows:
        group_key = (
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
                context=group_key[0] if split_flags["context"] else "",
            )
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=color,
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


def _unique_legend_entries(axes):
    """Collect de-duplicated legend entries from one or more axes."""
    entries = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_") and label not in entries:
                entries[label] = handle
    return list(entries.values()), list(entries.keys())


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

    handles, labels = _unique_legend_entries(axes)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), max(1, len(strategies))),
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.suptitle("Core Metric Sweep Overview")
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    _finalize_figure(fig, output_path)
    _write_rows(csv_path, csv_rows)
    return csv_rows


