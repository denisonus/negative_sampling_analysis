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
from scipy import stats

from models import TwoTowerModel
from samplers import get_sampler
from utils import (
    load_recbole_dataset,
    build_user_item_dict_from_train,
    compute_item_popularity_from_train,
    get_train_interactions,
    SimpleDataLoader,
    Trainer,
    InBatchTrainer,
)
from evaluation import Evaluator

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    """Get device based on config and availability."""
    device_setting = config.get("device", "auto")

    if device_setting == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
    elif device_setting == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_setting == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(config, sampling_strategy, device):
    """Run a single experiment with specified sampling strategy."""
    print(f"\n{'=' * 60}")
    print(f"Running experiment with {sampling_strategy} sampling")
    print(f"{'=' * 60}")

    # Load data
    print("Loading dataset...")
    recbole_config, dataset, train_data, valid_data, test_data = load_recbole_dataset(
        config["dataset"],
        config.get("data_path", "dataset/"),
        min_rating=config["min_rating"],
    )

    num_users = dataset.num(dataset.uid_field)
    num_items = dataset.num(dataset.iid_field)
    num_train = len(get_train_interactions(train_data))
    rating_filter = config["min_rating"]
    print(
        f"Dataset: {config['dataset']} | Users: {num_users}, Items: {num_items} | "
        f"rating >= {rating_filter}"
    )

    train_interactions = get_train_interactions(train_data)
    user_item_dict = build_user_item_dict_from_train(train_interactions)
    item_popularity = compute_item_popularity_from_train(train_interactions, num_items)
    print(f"Training interactions: {num_train}")

    train_loader = SimpleDataLoader(
        train_interactions,
        batch_size=config.get("train_batch_size", 1024),
        shuffle=True,
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_size=config.get("embedding_size", 64),
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
    )

    sampler = get_sampler(
        strategy=sampling_strategy,
        num_items=num_items,
        num_neg_samples=config.get("num_neg_samples", 4),
        user_item_dict=user_item_dict,
        item_popularity=item_popularity,
        model=model,
        device=device,
        candidate_pool_size=config.get("candidate_pool_size", 100),
        hard_ratio=config.get("hard_neg_ratio", 0.5),
        dns_temperature=config.get("dns_temperature", 0.1),
        curriculum_start_ratio=config.get("curriculum_start_ratio", 0.0),
        curriculum_end_ratio=config.get("curriculum_end_ratio", 0.8),
        curriculum_warmup_epochs=config.get("curriculum_warmup_epochs", 10),
        tau_plus=config.get("tau_plus", 0.05),
        smoothing=config.get("smoothing", 0.75),
        logq_correction=config.get("logq_correction", False),
    )

    evaluator = Evaluator(
        num_items=num_items,
        metrics=config.get("metrics", ["Recall", "NDCG", "MRR", "Hit"]),
        topk=config.get("topk", [5, 10, 20]),
        device=device,
    )

    if sampling_strategy == "in_batch":
        trainer = InBatchTrainer(
            model, sampler, config, device, item_popularity=item_popularity
        )
    else:
        trainer = Trainer(model, sampler, config, device)

    print("\nTraining...")
    train_history = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_data,
        evaluator=evaluator,
        epochs=config.get("epochs", 50),
    )

    print("\nFinal Test Evaluation...")
    test_metrics = evaluator.evaluate(model, test_data)

    print(f"\nTest Results for {sampling_strategy}:")
    for metric, value in sorted(test_metrics.items()):
        print(f"  {metric}: {value:.4f}")

    return {
        "strategy": sampling_strategy,
        "train_history": train_history,
        "test_metrics": test_metrics,
        "timing": {
            "total_time": train_history.get("total_time", 0),
            "total_sampling_time": train_history.get("total_sampling_time", 0),
            "total_training_time": train_history.get("total_training_time", 0),
        },
        "dataset_stats": {
            "num_users": num_users,
            "num_items": num_items,
            "num_train_interactions": num_train,
        },
    }


def run_all_experiments(config, strategies=None, num_runs=1):
    """Run experiments for all sampling strategies with multiple seeds for statistical significance."""
    if strategies is None:
        strategies = [
            "uniform",
            "popularity",
            "hard",
            "mixed",
            "in_batch",
            "dns",
            "curriculum",
            "debiased",
        ]

    device = get_device(config)
    print(f"Using device: {device}")

    # Setup seeds for multiple runs
    base_seed = config.get("seed", 42)
    seeds = [base_seed + i * 1000 for i in range(num_runs)]

    all_results = {strategy: [] for strategy in strategies}

    for run_idx, seed in enumerate(seeds):
        print(f"\n{'#' * 60}")
        print(f"RUN {run_idx + 1}/{len(seeds)} (seed={seed})")
        print(f"{'#' * 60}")

        for strategy in strategies:
            try:
                set_seed(seed)
                result = run_experiment(config, strategy, device)
                result["seed"] = seed
                result["run"] = run_idx
                all_results[strategy].append(result)
            except Exception as e:
                print(f"Error running {strategy} with seed {seed}: {e}")
                import traceback

                traceback.print_exc()

    return all_results


def compute_statistics(all_results):
    """Compute mean, std, and confidence intervals for metrics across runs."""
    stats_results = {}

    for strategy, runs in all_results.items():
        if not runs:
            continue

        # Collect all metrics across runs
        metrics_values = {}
        timing_values = {"total_time": [], "sampling_time": [], "training_time": []}

        for run in runs:
            for metric, value in run["test_metrics"].items():
                if metric not in metrics_values:
                    metrics_values[metric] = []
                metrics_values[metric].append(value)

            timing_values["total_time"].append(run["timing"]["total_time"])
            timing_values["sampling_time"].append(run["timing"]["total_sampling_time"])
            timing_values["training_time"].append(run["timing"]["total_training_time"])

        # Compute statistics
        stats_results[strategy] = {"metrics": {}, "timing": {}, "num_runs": len(runs)}

        for metric, values in metrics_values.items():
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)

            # 95% confidence interval
            if len(values) > 1:
                ci = stats.t.interval(
                    0.95, len(values) - 1, loc=mean, scale=stats.sem(values)
                )
                ci_lower, ci_upper = ci
            else:
                ci_lower, ci_upper = mean, mean

            stats_results[strategy]["metrics"][metric] = {
                "mean": float(mean),
                "std": float(std),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "values": [float(v) for v in values],
            }

        # Timing statistics
        for timing_key, values in timing_values.items():
            values = np.array(values)
            stats_results[strategy]["timing"][timing_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return stats_results


def save_results(all_results, output_dir="results", config=None):
    """Compare and save results with statistical analysis."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute statistics
    stats_results = compute_statistics(all_results)

    print("\n" + "=" * 100)
    print("COMPARISON OF NEGATIVE SAMPLING STRATEGIES (with statistical significance)")
    print("=" * 100)

    metrics_to_show = ["ndcg@10", "recall@10", "hit@10", "mrr@10"]

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
                    "timing": r["timing"],
                    "best_epoch": r["train_history"].get("best_epoch", -1),
                    "epochs_trained": len(r["train_history"].get("train_losses", [])),
                    "early_stopped": (
                        len(r["train_history"].get("train_losses", []))
                        < config.get("epochs", 50)
                        if config
                        else False
                    ),
                    "train_losses": r["train_history"].get("train_losses", []),
                    "valid_metrics": r["train_history"].get("valid_metrics", []),
                    "epoch_times": r["train_history"].get("epoch_times", []),
                    "sampling_times": r["train_history"].get("sampling_times", []),
                    "training_times": r["train_history"].get("training_times", []),
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
            "device": str(get_device(config)) if config else "unknown",
        },
    }
    metadata_file = os.path.join(run_output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

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
        help="Number of runs with different seeds for statistical significance",
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
