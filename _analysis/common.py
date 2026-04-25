"""Shared helpers for experiment analysis."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

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



def _sort_param_value(value):
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    if value is None:
        return (1, "")
    return (2, str(value))



def _load_metadata_for_results(results_file):
    """Load sibling metadata.json if it exists."""
    metadata_path = os.path.join(os.path.dirname(results_file), "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r") as metadata_file:
        return json.load(metadata_file)



