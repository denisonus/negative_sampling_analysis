"""Main Experiment Runner for comparing negative sampling strategies."""

import os
import sys
import platform
import yaml
import json
import torch
import numpy as np
import random
import argparse
import warnings
from datetime import datetime
from models import TwoTowerModel
from samplers import get_sampler
from utils import (
    load_recbole_dataset,
    extract_feature_data,
    build_user_item_dict_from_train,
    compute_item_popularity_from_train,
    compute_user_interaction_counts_from_train,
    get_train_interactions,
    TrainLoader,
    Trainer,
    InBatchTrainer,
    MixedInBatchTrainer,
)
from evaluation import Evaluator, compute_quality_metrics
from utils.experiment_config import COMMON_DEFAULTS, resolve_config

warnings.filterwarnings("ignore", category=FutureWarning)


def _metric_k(metric_name):
    if not metric_name or "@" not in metric_name:
        return None
    try:
        return int(str(metric_name).rsplit("@", 1)[1])
    except ValueError:
        return None


def _primary_k(config):
    valid_metric_k = _metric_k(config["valid_metric"])
    if valid_metric_k is not None:
        return valid_metric_k
    topk = config["topk"]
    return 10 if 10 in topk else min(topk)


def _metrics_to_show(config):
    k = _primary_k(config)
    metrics = [f"ndcg@{k}", f"recall@{k}", "recall@20", f"mrr@{k}", f"hit@{k}"]
    return list(dict.fromkeys(metrics))


def _quality_metrics_to_show(config):
    k = _primary_k(config)
    return [
        f"item_coverage@{k}",
        f"novelty@{k}",
        f"avg_popularity@{k}",
        f"personalization@{k}",
    ]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    return resolve_config(raw_config)


def prepare_experiment_data(config):
    """Load and precompute dataset objects shared by strategies in one run."""
    feature_aware = config["feature_aware"]
    implicit_feedback = config["implicit_feedback"]

    recbole_config, dataset, train_data, valid_data, test_data = load_recbole_dataset(
        config["dataset"],
        config["data_path"],
        min_rating=config["min_rating"],
        feature_aware=feature_aware,
        implicit_feedback=implicit_feedback,
        benchmark_filename=config.get("benchmark_filename"),
        metrics=config["metrics"],
        topk=config["topk"],
        valid_metric=config["valid_metric"],
    )

    feature_data = None
    if feature_aware:
        feature_data = extract_feature_data(dataset, config["dataset"])

    num_items = dataset.num(dataset.iid_field)
    train_interactions = get_train_interactions(train_data)

    return {
        "recbole_config": recbole_config,
        "dataset": dataset,
        "train_data": train_data,
        "valid_data": valid_data,
        "test_data": test_data,
        "feature_data": feature_data,
        "train_interactions": train_interactions,
        "user_item_dict": build_user_item_dict_from_train(train_interactions),
        "item_popularity": compute_item_popularity_from_train(
            train_interactions, num_items
        ),
        "user_train_counts": compute_user_interaction_counts_from_train(
            train_interactions
        ),
    }


def run_experiment(config, sampling_strategy, device, seed=None, prepared_data=None):
    """Run a single experiment with specified sampling strategy."""
    print(f"\n{'=' * 60}")
    print(f"Running experiment with {sampling_strategy} sampling")
    print(f"{'=' * 60}")
    feature_aware = config["feature_aware"]
    implicit_feedback = config["implicit_feedback"]
    min_rating = config["min_rating"]

    if prepared_data is None:
        print("Loading dataset...")
        prepared_data = prepare_experiment_data(config)
    else:
        print("Using prepared dataset...")

    dataset = prepared_data["dataset"]
    valid_data = prepared_data["valid_data"]
    test_data = prepared_data["test_data"]
    feature_data = prepared_data["feature_data"]
    train_interactions = prepared_data["train_interactions"]
    user_item_dict = prepared_data["user_item_dict"]
    item_popularity = prepared_data["item_popularity"]
    user_train_counts = prepared_data["user_train_counts"]

    num_users = dataset.num(dataset.uid_field)
    num_items = dataset.num(dataset.iid_field)
    num_train = len(train_interactions)
    feedback_label = (
        "implicit feedback"
        if implicit_feedback
        else f"rating >= {min_rating}"
    )
    print(
        f"Dataset: {config['dataset']} | Users: {num_users}, Items: {num_items} | "
        f"{feedback_label}"
    )
    print(f"Feature-aware mode: {'on' if feature_aware else 'off'}")
    if feature_data is not None:
        user_feature_names = [spec["name"] for spec in feature_data["user"]["schema"]]
        item_feature_names = [spec["name"] for spec in feature_data["item"]["schema"]]
        print(
            f"User features: {user_feature_names or 'none'} | "
            f"Item features: {item_feature_names or 'none'}"
        )

    print(f"Training interactions: {num_train}")

    train_loader = TrainLoader(
        train_interactions,
        batch_size=config["train_batch_size"],
        shuffle=True,
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_size=config["embedding_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        user_feature_schema=(
            feature_data["user"]["schema"] if feature_data is not None else None
        ),
        user_feature_tensors=(
            feature_data["user"]["tensors"] if feature_data is not None else None
        ),
        item_feature_schema=(
            feature_data["item"]["schema"] if feature_data is not None else None
        ),
        item_feature_tensors=(
            feature_data["item"]["tensors"] if feature_data is not None else None
        ),
    )

    sampler = get_sampler(
        strategy=sampling_strategy,
        num_items=num_items,
        num_neg_samples=config["num_neg_samples"],
        user_item_dict=user_item_dict,
        item_popularity=item_popularity,
        model=model,
        device=device,
        candidate_pool_size=config["candidate_pool_size"],
        hard_ratio=config["hard_neg_ratio"],
        dns_temperature=config["dns_temperature"],
        curriculum_start_ratio=config["curriculum_start_ratio"],
        curriculum_end_ratio=config["curriculum_end_ratio"],
        curriculum_warmup_epochs=config["curriculum_warmup_epochs"],
        tau_plus=config["tau_plus"],
        smoothing=config["smoothing"],
        logq_correction=config["logq_correction"],
        train_batch_size=config["train_batch_size"],
        mixed_index_batch_size=config["mixed_index_batch_size"],
    )
    print(f"Resolved sampler: {sampler.name}")

    evaluator = Evaluator(
        num_items=num_items,
        metrics=config["metrics"],
        topk=config["topk"],
        device=device,
        batch_size=config["eval_batch_size"],
    )

    if sampler.name == "in_batch":
        trainer = InBatchTrainer(
            model, sampler, config, device, item_popularity=item_popularity
        )
    elif sampler.name == "mixed_in_batch_uniform":
        trainer = MixedInBatchTrainer(
            model, sampler, config, device, item_popularity=item_popularity
        )
    else:
        trainer = Trainer(model, sampler, config, device)

    print("\nTraining...")
    train_history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_data,
        evaluator=evaluator,
        epochs=config["epochs"],
    )

    print("\nFinal Test Evaluation...")
    test_rankings = evaluator.rank(model, test_data)
    test_metrics = evaluator.evaluate_from_rankings(test_rankings)
    bucket_k = 10 if 10 in evaluator.topk else _primary_k(config)
    bucket_metrics = evaluator.evaluate_user_buckets_from_rankings(
        test_rankings,
        user_train_counts=user_train_counts,
        target_k=bucket_k,
    )
    if not bucket_metrics and bucket_k not in evaluator.topk:
        print(
            f"Warning: user bucket metrics skipped because {bucket_k} is not in configured topk"
        )
    quality_metrics = compute_quality_metrics(
        test_rankings["topk_items"],
        item_popularity=item_popularity,
        num_items=num_items,
        topk=config["topk"],
        seed=config["seed"] if seed is None else seed,
    )

    print(f"\nTest Results for {sampling_strategy}:")
    for metric, value in sorted(test_metrics.items()):
        print(f"  {metric}: {value:.4f}")
    print("Quality Metrics:")
    for metric, value in sorted(quality_metrics.items()):
        print(f"  {metric}: {value:.4f}")

    return {
        "strategy": sampler.name,
        "train_history": train_history,
        "test_metrics": test_metrics,
        "bucket_metrics": bucket_metrics,
        "quality_metrics": quality_metrics,
        "timing": {
            "total_time": train_history.get("total_time", 0),
            "total_sampling_time": train_history.get("total_sampling_time", 0),
            "total_training_time": train_history.get("total_training_time", 0),
        },
        "dataset_stats": {
            "num_users": num_users,
            "num_items": num_items,
            "num_train_interactions": num_train,
            "feature_aware": feature_aware,
            "implicit_feedback": implicit_feedback,
        },
    }


def run_all_experiments(config, strategies=None, num_runs=1):
    """Run experiments for all sampling strategies with multiple seeds."""
    if strategies is None:
        strategies = [
            "uniform",
            "popularity",
            "hard",
            "mixed_hard_uniform",
            "mixed_in_batch_uniform",
            "in_batch",
            "dns",
            "curriculum",
            "debiased",
        ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup seeds for multiple runs
    base_seed = config["seed"]
    seeds = [base_seed + i * 1000 for i in range(num_runs)]

    all_results = {strategy: [] for strategy in strategies}

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'#' * 60}")
        print(f"RUN {run_idx + 1}/{len(seeds)} (seed={seed})")
        print(f"{'#' * 60}")

        set_seed(seed)
        print("Preparing shared dataset...")
        prepared_data = prepare_experiment_data(config)

        for strategy in strategies:
            try:
                set_seed(seed)
                result = run_experiment(
                    config,
                    strategy,
                    device,
                    seed=seed,
                    prepared_data=prepared_data,
                )
                result["seed"] = seed
                result["run"] = run_idx
                all_results[strategy].append(result)
            except Exception as e:
                print(f"Error running {strategy} with seed {seed}: {e}")
                import traceback

                traceback.print_exc()

    return all_results


def compute_statistics(all_results):
    """Compute mean and std for metrics across runs."""
    stats_results = {}

    def summarize_values(values):
        values = np.array(values, dtype=np.float64)
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": [float(v) for v in values],
        }

    for strategy, runs in all_results.items():
        if not runs:
            continue

        # Collect all metrics across runs
        metrics_values = {}
        bucket_values = {}
        quality_values = {}
        timing_values = {"total_time": [], "sampling_time": [], "training_time": []}

        for run in runs:
            for metric, value in run["test_metrics"].items():
                if metric not in metrics_values:
                    metrics_values[metric] = []
                metrics_values[metric].append(value)

            for bucket_label, bucket_metrics in run.get("bucket_metrics", {}).items():
                if bucket_label not in bucket_values:
                    bucket_values[bucket_label] = {}
                for metric, value in bucket_metrics.items():
                    if metric not in bucket_values[bucket_label]:
                        bucket_values[bucket_label][metric] = []
                    bucket_values[bucket_label][metric].append(value)

            for metric, value in run.get("quality_metrics", {}).items():
                if metric not in quality_values:
                    quality_values[metric] = []
                quality_values[metric].append(value)

            timing_values["total_time"].append(run["timing"]["total_time"])
            timing_values["sampling_time"].append(run["timing"]["total_sampling_time"])
            timing_values["training_time"].append(run["timing"]["total_training_time"])

        # Compute statistics
        stats_results[strategy] = {
            "metrics": {},
            "bucket_metrics": {},
            "quality_metrics": {},
            "timing": {},
            "num_runs": len(runs),
        }

        for metric, values in metrics_values.items():
            stats_results[strategy]["metrics"][metric] = summarize_values(values)

        for bucket_label, metric_values in bucket_values.items():
            stats_results[strategy]["bucket_metrics"][bucket_label] = {}
            for metric, values in metric_values.items():
                stats_results[strategy]["bucket_metrics"][bucket_label][metric] = (
                    summarize_values(values)
                )

        for metric, values in quality_values.items():
            stats_results[strategy]["quality_metrics"][metric] = summarize_values(values)

        # Timing statistics
        for timing_key, values in timing_values.items():
            values = np.array(values)
            stats_results[strategy]["timing"][timing_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return stats_results


def save_results(all_results, output_dir="results", config=None):
    """Compare and save results with summary statistics."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute statistics
    stats_results = compute_statistics(all_results)

    print("\n" + "=" * 100)
    print("COMPARISON OF NEGATIVE SAMPLING STRATEGIES (summary statistics)")
    print("=" * 100)

    display_config = config or COMMON_DEFAULTS
    metrics_to_show = _metrics_to_show(display_config)
    quality_metrics_to_show = _quality_metrics_to_show(display_config)

    # Header
    header = f"{'Strategy':<15}"
    for m in metrics_to_show:
        header += f"{m:<20}"
    header += f"{'Time (s)':<15}{'Samp. Time':<15}"
    print(header)
    print("-" * 100)

    for strategy, stats_data in stats_results.items():
        row = f"{strategy:<15}"
        for metric in metrics_to_show:
            if metric in stats_data["metrics"]:
                m = stats_data["metrics"][metric]
                row += f"{m['mean']:.4f}±{m['std']:.4f}   "
            else:
                row += f"{'N/A':<20}"

        timing = stats_data["timing"]
        row += f"{timing['total_time']['mean']:.1f}±{timing['total_time']['std']:.1f}".ljust(
            15
        )
        row += f"{timing['sampling_time']['mean']:.1f}±{timing['sampling_time']['std']:.1f}".ljust(
            15
        )
        print(row)

    if any(stats_data["quality_metrics"] for stats_data in stats_results.values()):
        print("\n" + "=" * 100)
        print(f"RECOMMENDATION QUALITY METRICS (@{_primary_k(display_config)})")
        print("=" * 100)
        header = f"{'Strategy':<15}"
        for metric in quality_metrics_to_show:
            header += f"{metric:<24}"
        print(header)
        print("-" * 100)

        for strategy, stats_data in stats_results.items():
            row = f"{strategy:<15}"
            for metric in quality_metrics_to_show:
                if metric in stats_data["quality_metrics"]:
                    m = stats_data["quality_metrics"][metric]
                    row += f"{m['mean']:.4f}±{m['std']:.4f}   "
                else:
                    row += f"{'N/A':<24}"
            print(row)

    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped folder for this experiment run
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(run_output_dir, "results.json")
    save_data = {
        "statistics": stats_results,
        "raw_results": {
            strategy: [
                {
                    "seed": r["seed"],
                    "test_metrics": r["test_metrics"],
                    "bucket_metrics": r.get("bucket_metrics", {}),
                    "quality_metrics": r.get("quality_metrics", {}),
                    "timing": r["timing"],
                    "best_epoch": r["train_history"].get("best_epoch", -1),
                    "epochs_trained": len(r["train_history"].get("train_losses", [])),
                    "early_stopped": (
                        len(r["train_history"].get("train_losses", []))
                        < config["epochs"]
                        if config
                        else False
                    ),
                    "train_losses": r["train_history"].get("train_losses", []),
                    "valid_metrics": r["train_history"].get("valid_metrics", []),
                }
                for r in runs
            ]
            for strategy, runs in all_results.items()
        },
    }

    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)

    # Collect dataset stats from the first available run
    dataset_stats = {}
    for runs in all_results.values():
        if runs and "dataset_stats" in runs[0]:
            dataset_stats = runs[0]["dataset_stats"]
            break

    # Save a metadata file with run information
    metadata = {
        "timestamp": timestamp,
        "strategies": list(all_results.keys()),
        "num_runs": len(list(all_results.values())[0]) if all_results else 0,
        "results_file": "results.json",
        "config": config,
        "dataset_stats": dataset_stats,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        },
    }
    metadata_file = os.path.join(run_output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    try:
        from analysis import generate_full_report

        generate_full_report(results_file, output_dir=run_output_dir)
    except Exception as e:
        print(f"Warning: automatic analysis generation failed: {e}")

    print(f"\nResults saved to: {run_output_dir}")
    return stats_results, run_output_dir


def main():
    parser = argparse.ArgumentParser(description="Run negative sampling experiments")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=None,
        help="Sampling strategies to evaluate",
    )
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of runs with different seeds",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    all_results = run_all_experiments(
        config, strategies=args.strategies, num_runs=args.num_runs
    )

    if all_results:
        stats_results, output_dir = save_results(
            all_results, args.output, config=config
        )
        print(f"\nExperiment complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
