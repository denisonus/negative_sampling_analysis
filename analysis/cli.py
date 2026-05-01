"""Command-line entry point for experiment analysis."""

import argparse
import os

import matplotlib.pyplot as plt

from .common import DEFAULT_SWEEP_METRICS
from .report import generate_full_report
from .sweeps import (
    plot_feature_uplift,
    plot_multi_metric_sweep,
    save_feature_uplift_table,
)


def main():
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

    if len(args.results_files) == 1 and args.sweep_strategy is None and args.sweep_strategies is None:
        generate_full_report(args.results_files[0], args.output_dir)
        return

    if args.sweep_strategy and args.sweep_strategies:
        raise SystemExit("Use either --sweep_strategy or --sweep_strategies, not both.")
    if args.sweep_strategy is None and args.sweep_strategies is None:
        raise SystemExit("Multiple results files require --sweep_strategy or --sweep_strategies.")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.results_files[0]), "sweep_analysis"
    )
    os.makedirs(output_dir, exist_ok=True)
    plt.switch_backend("Agg")

    strategies = args.sweep_strategies or [args.sweep_strategy]
    plot_multi_metric_sweep(
        args.results_files,
        strategies=strategies,
        metrics=DEFAULT_SWEEP_METRICS,
        param=args.sweep_param,
        output_path=os.path.join(output_dir, "parameter_sweep_core_metrics.png"),
        csv_path=os.path.join(output_dir, "parameter_sweep_core_metrics.csv"),
    )

    save_feature_uplift_table(
        args.results_files,
        os.path.join(output_dir, "feature_uplift.csv"),
        strategies=strategies,
        param=args.sweep_param,
    )
    plot_feature_uplift(
        args.results_files,
        os.path.join(output_dir, "feature_uplift.png"),
        strategies=strategies,
        param=args.sweep_param,
    )


if __name__ == "__main__":
    main()
