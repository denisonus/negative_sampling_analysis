"""Lean report generation for completed experiment runs."""

import os
from datetime import datetime

import matplotlib.pyplot as plt

from .common import (
    _load_metadata_for_results,
    _preferred_metric,
    _title_suffix_from_metadata,
    load_results,
)
from .plots import (
    plot_ablation_delta,
    plot_metric_by_k,
    plot_quality_small_multiples,
    plot_thesis_dashboard,
    plot_training_dynamics,
    plot_user_bucket_delta_heatmap,
)
from .tables import (
    save_relative_improvement_table,
    save_significance_table,
    save_summary_table,
    save_user_bucket_metrics_table,
)


def generate_full_report(results_file, output_dir=None):
    """Generate a compact thesis-oriented analysis bundle."""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"analysis_{timestamp}")

    plt.switch_backend("Agg")
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(results_file)
    metadata = _load_metadata_for_results(results_file)
    title_suffix = _title_suffix_from_metadata(metadata)
    stats = results.get("statistics", {})
    primary_metric = _preferred_metric(stats, metric_base="ndcg")
    primary_metric_file_label = primary_metric.replace("@", "")

    save_summary_table(
        results, os.path.join(output_dir, "summary_metrics.csv"), metadata=metadata
    )
    save_relative_improvement_table(
        results,
        os.path.join(output_dir, "relative_improvements_vs_uniform.csv"),
        baseline="uniform",
        metadata=metadata,
    )
    plot_thesis_dashboard(
        results,
        output_path=os.path.join(output_dir, "dashboard.png"),
        title_suffix=title_suffix,
    )
    plot_metric_by_k(
        results,
        output_path=os.path.join(output_dir, "metric_by_k.png"),
        title_suffix=title_suffix,
    )

    has_quality_metrics = any(
        strategy_stats.get("quality_metrics") for strategy_stats in stats.values()
    )
    if has_quality_metrics:
        plot_quality_small_multiples(
            results,
            output_path=os.path.join(output_dir, "quality_metrics.png"),
            title_suffix=title_suffix,
        )

    has_bucket_metrics = any(
        strategy_stats.get("bucket_metrics") for strategy_stats in stats.values()
    )
    if has_bucket_metrics:
        save_user_bucket_metrics_table(
            results, os.path.join(output_dir, "user_bucket_metrics.csv")
        )
        plot_user_bucket_delta_heatmap(
            results,
            baseline="uniform",
            metric=primary_metric,
            output_path=os.path.join(
                output_dir, f"user_bucket_{primary_metric_file_label}_delta_heatmap.png"
            ),
            title_suffix=title_suffix,
        )

    if "uniform" in stats:
        plot_ablation_delta(
            results,
            baseline="uniform",
            metric=primary_metric,
            output_path=os.path.join(
                output_dir, f"{primary_metric_file_label}_delta_vs_uniform.png"
            ),
            title_suffix=title_suffix,
        )

    if "raw_results" in results:
        plot_training_dynamics(
            results,
            target_metric=primary_metric,
            output_path=os.path.join(output_dir, "training_dynamics.png"),
            title_suffix=title_suffix,
        )

    save_significance_table(
        results,
        os.path.join(output_dir, f"significance_{primary_metric_file_label}.csv"),
        metric=primary_metric,
    )

    print(f"Analysis bundle saved to: {output_dir}")
