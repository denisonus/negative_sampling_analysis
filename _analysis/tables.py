"""CSV table generation and significance tests."""

import csv

import numpy as np
from scipy import stats

from .common import (
    DEFAULT_BUCKET_METRICS,
    DEFAULT_QUALITY_METRICS,
    DEFAULT_RELEVANCE_METRICS,
    SUMMARY_OPTIONAL_RELEVANCE_METRICS,
    _available_bucket_metrics,
    _available_metrics,
    _collect_bucket_labels,
    _feature_aware_from_metadata,
    _get_metric_value,
    _relative_improvement,
    _sorted_strategies,
)

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
        ["item_coverage@10", "novelty@10", "avg_popularity@10", "personalization@10"],
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


def _relative_metric_specs(stats_data, metrics=None):
    """Build metric specs for relative improvement tables."""
    if metrics is not None:
        specs = []
        for metric in metrics:
            if any(metric in stats.get("metrics", {}) for stats in stats_data.values()):
                specs.append(("relevance", "metrics", metric))
            elif any(
                metric in stats.get("quality_metrics", {})
                for stats in stats_data.values()
            ):
                specs.append(("quality", "quality_metrics", metric))
        return specs

    all_relevance = [*DEFAULT_RELEVANCE_METRICS, *SUMMARY_OPTIONAL_RELEVANCE_METRICS]
    relevance = _available_metrics(stats_data, all_relevance)
    quality = _available_metrics(stats_data, DEFAULT_QUALITY_METRICS, section="quality_metrics")
    return [
        *[("relevance", "metrics", m) for m in relevance],
        *[("quality", "quality_metrics", m) for m in quality],
    ]


def build_relative_improvement_rows(
    results, baseline="uniform", metrics=None, metadata=None
):
    """Build per-strategy metric improvements relative to a baseline."""
    stats_data = results.get("statistics", {})
    if baseline not in stats_data:
        return []

    metric_specs = _relative_metric_specs(stats_data, metrics=metrics)
    if not metric_specs:
        return []

    rows = []
    feature_aware = _feature_aware_from_metadata(metadata)
    strategies = [
        strategy
        for strategy in _sorted_strategies(stats_data)
        if strategy != baseline
    ]

    for strategy in strategies:
        strategy_stats = stats_data[strategy]
        for metric_type, section, metric in metric_specs:
            baseline_stats = stats_data[baseline].get(section, {}).get(metric)
            strategy_metric_stats = strategy_stats.get(section, {}).get(metric)
            if baseline_stats is None or strategy_metric_stats is None:
                continue

            baseline_value = baseline_stats.get("mean", 0.0)
            strategy_value = strategy_metric_stats.get("mean", 0.0)
            absolute_delta = strategy_value - baseline_value
            relative_value = _relative_improvement(strategy_value, baseline_value)
            rows.append(
                {
                    "baseline": baseline,
                    "strategy": strategy,
                    "metric_type": metric_type,
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "strategy_value": strategy_value,
                    "absolute_delta": absolute_delta,
                    "relative_improvement": relative_value,
                    "relative_improvement_percent": (
                        None if relative_value is None else relative_value * 100
                    ),
                    "feature_aware": feature_aware,
                }
            )

    return rows


def save_relative_improvement_table(
    results, output_path, baseline="uniform", metrics=None, metadata=None
):
    """Persist percentage metric improvements relative to a baseline."""
    rows = build_relative_improvement_rows(
        results, baseline=baseline, metrics=metrics, metadata=metadata
    )
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



