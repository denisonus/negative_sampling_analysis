"""Post-hoc recommendation-quality metrics."""

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _default_quality_results(topk: Iterable[int]) -> Dict[str, float]:
    metric_names = (
        "item_coverage",
        "novelty",
        "avg_popularity",
        "personalization",
    )
    return {
        f"{metric_name}@{k}": 0.0
        for metric_name in metric_names
        for k in sorted({int(k) for k in topk})
    }


def _sample_user_pairs(
    num_users: int, max_pairs: int, seed: int
) -> List[Tuple[int, int]]:
    """Deterministically sample up to ``max_pairs`` unique user pairs."""
    total_pairs = num_users * (num_users - 1) // 2
    if total_pairs <= 0:
        return []

    if total_pairs <= max_pairs:
        return list(combinations(range(num_users), 2))

    rng = np.random.default_rng(seed)
    sampled_pairs = set()
    while len(sampled_pairs) < max_pairs:
        user_a, user_b = rng.choice(num_users, size=2, replace=False)
        if user_a > user_b:
            user_a, user_b = user_b, user_a
        sampled_pairs.add((int(user_a), int(user_b)))

    return sorted(sampled_pairs)


def _compute_personalization(
    recommendations: np.ndarray, seed: int, pair_sample_cap: int
) -> float:
    """Compute 1 - mean Jaccard similarity over user recommendation sets."""
    num_users = recommendations.shape[0]
    if num_users < 2:
        return 0.0

    user_sets = [set(row.tolist()) for row in recommendations]
    pairs = _sample_user_pairs(num_users, pair_sample_cap, seed)
    if not pairs:
        return 0.0

    similarities = []
    for user_a, user_b in pairs:
        rec_a = user_sets[user_a]
        rec_b = user_sets[user_b]
        union_size = len(rec_a | rec_b)
        if union_size == 0:
            similarities.append(0.0)
            continue
        similarities.append(len(rec_a & rec_b) / union_size)

    return float(1.0 - np.mean(similarities))


def compute_quality_metrics(
    topk_items: np.ndarray,
    item_popularity: np.ndarray,
    num_items: int,
    topk: Iterable[int],
    seed: int = 42,
    pair_sample_cap: int = 10_000,
) -> Dict[str, float]:
    """Compute recommendation-quality metrics from final top-k rankings."""
    results = _default_quality_results(topk)
    topk_items = np.asarray(topk_items, dtype=np.int64)

    if num_items <= 0 or topk_items.size == 0:
        return results

    popularity = np.asarray(item_popularity, dtype=np.float64)
    if popularity.size < num_items:
        popularity = np.pad(
            popularity, (0, num_items - popularity.size), constant_values=1.0
        )
    else:
        popularity = popularity[:num_items]

    popularity = np.clip(popularity, 1e-12, None)
    item_probabilities = popularity / popularity.sum()

    max_available_k = topk_items.shape[1]
    for k in sorted({int(k) for k in topk}):
        effective_k = min(k, max_available_k)
        if effective_k <= 0:
            continue

        recommendations = topk_items[:, :effective_k]
        unique_items = np.unique(recommendations)
        results[f"item_coverage@{k}"] = float(unique_items.size / num_items)
        results[f"novelty@{k}"] = float(
            (-np.log2(item_probabilities[recommendations])).mean()
        )
        results[f"avg_popularity@{k}"] = float(popularity[recommendations].mean())
        results[f"personalization@{k}"] = _compute_personalization(
            recommendations, seed=seed + k, pair_sample_cap=pair_sample_cap
        )

    return results
