"""Experiment configuration defaults and validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


COMMON_DEFAULTS = {
    "seed": 42,
    "data_path": "dataset/",
    "embedding_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "feature_aware": False,
    "epochs": 50,
    "train_batch_size": 512,
    "learning_rate": 0.0003,
    "weight_decay": 0.0005,
    "patience": 8,
    "num_neg_samples": 4,
    "candidate_pool_size": 100,
    "hard_neg_ratio": 0.5,
    "dns_temperature": 0.1,
    "curriculum_start_ratio": 0.0,
    "curriculum_end_ratio": 0.8,
    "curriculum_warmup_epochs": 8,
    "tau_plus": 0.05,
    "smoothing": 0.75,
    "logq_correction": True,
    "mixed_index_batch_size": 512,
    "eval_batch_size": 256,
    "metrics": ["Recall", "NDCG", "MRR", "Hit"],
    "topk": [5, 10, 20],
    "valid_metric": "NDCG@10",
}


DATASET_PRESETS = {
    "ml-100k": {
        "implicit_feedback": False,
        "min_rating": 4,
    },
    "gowalla-1m": {
        "implicit_feedback": True,
        "benchmark_filename": ["train", "valid", "test"],
        "min_rating": None,
        "epochs": 30,
        "train_batch_size": 1024,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 5,
        "candidate_pool_size": 300,
        "mixed_index_batch_size": 1024,
        "eval_batch_size": 512,
        "curriculum_warmup_epochs": 10,
        "metrics": ["Recall", "NDCG", "MRR", "Hit"],
        "topk": [20, 50],
        "valid_metric": "NDCG@20",
    },
}


ALLOWED_CONFIG_KEYS = frozenset(
    {"dataset"}
    | set(COMMON_DEFAULTS)
    | {key for preset in DATASET_PRESETS.values() for key in preset}
)


def resolve_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve a compact experiment config into the full runtime config."""
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config must be a YAML mapping")

    unknown_keys = sorted(set(raw_config) - ALLOWED_CONFIG_KEYS)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unknown config key(s): {joined}")

    dataset = raw_config.get("dataset")
    if dataset is None:
        raise ValueError("Config must define a dataset")
    if dataset not in DATASET_PRESETS:
        supported = ", ".join(sorted(DATASET_PRESETS))
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported datasets: {supported}"
        )

    resolved = deepcopy(COMMON_DEFAULTS)
    resolved.update(deepcopy(DATASET_PRESETS[dataset]))
    resolved.update(deepcopy(raw_config))
    return resolved
