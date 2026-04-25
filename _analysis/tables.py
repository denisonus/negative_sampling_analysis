"""CSV table generation and significance tests."""

import csv

import numpy as np
from scipy import stats

from .common import (
    DEFAULT_BUCKET_METRICS,
    SUMMARY_OPTIONAL_RELEVANCE_METRICS,
    _available_bucket_metrics,
    _available_metrics,
    _collect_bucket_labels,
    _feature_aware_from_metadata,
    _get_metric_value,
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



