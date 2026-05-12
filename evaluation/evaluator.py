"""Evaluation utilities for top-k recommendation metrics."""

import numpy as np

import torch
from collections import defaultdict

from utils.experiment_config import COMMON_DEFAULTS


def _hit(pos_index, pos_len=None):
    return (np.cumsum(pos_index, axis=1) > 0).astype(np.float64)


def _mrr(pos_index, pos_len=None):
    first_hit = pos_index.argmax(axis=1)
    result = np.zeros_like(pos_index, dtype=np.float64)
    for row, idx in enumerate(first_hit):
        if pos_index[row, idx]:
            result[row, idx:] = 1.0 / (idx + 1)
    return result


def _recall(pos_index, pos_len):
    return np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)


def _ndcg(pos_index, pos_len):
    ranks = np.arange(1, pos_index.shape[1] + 1, dtype=np.float64)
    discounts = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, discounts, 0.0), axis=1)

    idcg = np.cumsum(np.broadcast_to(discounts, pos_index.shape), axis=1)
    idcg_len = np.minimum(pos_len, pos_index.shape[1])
    for row, length in enumerate(idcg_len):
        if length <= 0:
            idcg[row] = 1.0
        elif length < pos_index.shape[1]:
            idcg[row, length:] = idcg[row, length - 1]

    return dcg / idcg


class Evaluator:
    """Evaluator for recommendation models using local top-k metrics."""

    METRIC_FUNCTIONS = {
        "hit": _hit,
        "recall": _recall,
        "ndcg": _ndcg,
        "mrr": _mrr,
    }
    ACTIVITY_BUCKET_ORDER = ("0", "1-5", "6-20", "21+")

    def __init__(
        self,
        num_items,
        metrics=None,
        topk=None,
        device="cpu",
        batch_size=256,
    ):
        if metrics is None:
            metrics = COMMON_DEFAULTS["metrics"]
        if topk is None:
            topk = COMMON_DEFAULTS["topk"]

        self.num_items = num_items
        self.metrics = [m.lower() for m in metrics]
        unsupported_metrics = sorted(set(self.metrics) - set(self.METRIC_FUNCTIONS))
        if unsupported_metrics:
            raise ValueError(f"Unsupported metric(s): {', '.join(unsupported_metrics)}")
        self.topk = list(topk)
        self.max_k = max(self.topk)
        self.device = device
        self.batch_size = max(int(batch_size), 1)

    def _empty_results(self):
        return {f"{m}@{k}": 0.0 for m in self.metrics for k in self.topk}

    def _empty_rankings(self):
        return {
            "user_ids": np.array([], dtype=np.int64),
            "topk_items": np.empty((0, self.max_k), dtype=np.int64),
            "pos_index": np.empty((0, self.max_k), dtype=bool),
            "pos_len": np.array([], dtype=np.int64),
        }

    @classmethod
    def _activity_bucket_label(cls, interaction_count):
        if interaction_count <= 0:
            return "0"
        if interaction_count <= 5:
            return "1-5"
        if interaction_count <= 20:
            return "6-20"
        return "21+"

    @classmethod
    def _activity_bucket_sort_key(cls, label):
        try:
            return cls.ACTIVITY_BUCKET_ORDER.index(label)
        except ValueError:
            return len(cls.ACTIVITY_BUCKET_ORDER)

    def _metric_values_from_rankings(self, rankings, metric_name):
        pos_index = rankings["pos_index"]
        pos_len = rankings["pos_len"]

        if pos_index.size == 0:
            return np.empty((0, self.max_k), dtype=np.float64)

        metric_values = self.METRIC_FUNCTIONS[metric_name](pos_index, pos_len)

        return np.asarray(metric_values, dtype=np.float64)

    def _extract_interactions(self, test_data):
        test_user_items = {}
        uid2pos = test_data.uid2positive_item

        for uid in test_data.uid_list.numpy():
            test_user_items[int(uid)] = set(uid2pos[uid].tolist())

        return test_user_items

    def rank(self, model, test_data):
        """Return masked top-k rankings and hit indicators for evaluation."""
        model.eval()
        test_user_items = self._extract_interactions(test_data)

        if not test_user_items:
            return self._empty_rankings()

        uid2history = (
            test_data.uid2history_item if hasattr(test_data, "uid2history_item") else {}
        )

        users_list = list(test_user_items.keys())
        batch_size = self.batch_size
        all_user_ids, all_topk_items, all_pos_index, all_pos_len = [], [], [], []

        with torch.no_grad():
            # Pre-compute all item embeddings once
            all_item_emb = model.get_all_item_embeddings()

            for start_idx in range(0, len(users_list), batch_size):
                end_idx = min(start_idx + batch_size, len(users_list))
                batch_users = users_list[start_idx:end_idx]
                user_ids = torch.tensor(
                    batch_users, dtype=torch.long, device=self.device
                )

                scores = model.predict(user_ids, all_item_emb=all_item_emb)

                # Mask training history items
                for i, user in enumerate(batch_users):
                    history_items = uid2history[user] if user < len(uid2history) else []
                    if hasattr(history_items, "__len__") and len(history_items) > 0:
                        mask_indices = torch.tensor(
                            [
                                int(item)
                                for item in history_items
                                if int(item) < scores.size(1)
                            ],
                            dtype=torch.long,
                            device=scores.device,
                        )
                        if len(mask_indices) > 0:
                            scores[i, mask_indices] = float("-inf")

                _, topk_idx = torch.topk(
                    scores, k=min(self.max_k, scores.size(1)), dim=1
                )
                topk_idx = topk_idx.cpu().numpy()

                for i, user in enumerate(batch_users):
                    ground_truth = test_user_items[user]
                    if ground_truth:
                        all_user_ids.append(int(user))
                        all_topk_items.append(topk_idx[i])
                        pos_row = np.array(
                            [item in ground_truth for item in topk_idx[i]]
                        )
                        all_pos_index.append(pos_row)
                        all_pos_len.append(len(ground_truth))

        if not all_pos_index:
            return self._empty_rankings()

        return {
            "user_ids": np.array(all_user_ids, dtype=np.int64),
            "topk_items": np.array(all_topk_items, dtype=np.int64),
            "pos_index": np.array(all_pos_index),
            "pos_len": np.array(all_pos_len),
        }

    def evaluate_from_rankings(self, rankings):
        """Compute ranking metrics from precomputed top-k rankings."""
        pos_index = rankings["pos_index"]
        pos_len = rankings["pos_len"]

        if pos_index.size == 0:
            return self._empty_results()

        results = {}
        for metric_name in self.metrics:
            metric_values = self.METRIC_FUNCTIONS[metric_name](pos_index, pos_len)

            avg_values = metric_values.mean(axis=0)
            for k in self.topk:
                idx = k - 1 if k <= len(avg_values) else -1
                results[f"{metric_name}@{k}"] = float(avg_values[idx])

        return results

    def evaluate_user_buckets_from_rankings(
        self,
        rankings,
        user_train_counts,
        metrics=("ndcg", "recall", "hit", "mrr"),
        target_k=10,
    ):
        """Aggregate selected metrics by user activity bucket."""
        if target_k not in self.topk:
            return {}

        user_ids = np.asarray(rankings["user_ids"], dtype=np.int64)
        if user_ids.size == 0:
            return {}

        metric_index = target_k - 1
        bucketed_values = defaultdict(lambda: defaultdict(list))

        for metric_name in metrics:
            metric_values = self._metric_values_from_rankings(rankings, metric_name)
            if metric_values.ndim != 2 or metric_values.shape[0] != user_ids.size:
                continue
            if metric_index >= metric_values.shape[1]:
                continue

            metric_label = f"{metric_name}@{target_k}"
            per_user_values = metric_values[:, metric_index]
            for user_id, value in zip(user_ids, per_user_values):
                interaction_count = int(user_train_counts.get(int(user_id), 0))
                bucket_label = self._activity_bucket_label(interaction_count)
                bucketed_values[bucket_label][metric_label].append(float(value))

        results = {}
        for bucket_label in sorted(
            bucketed_values.keys(), key=self._activity_bucket_sort_key
        ):
            results[bucket_label] = {}
            for metric_label, values in bucketed_values[bucket_label].items():
                if values:
                    results[bucket_label][metric_label] = float(
                        np.mean(np.asarray(values, dtype=np.float64))
                    )

        return results

    def evaluate(self, model, test_data):
        """Evaluate model on test data."""
        return self.evaluate_from_rankings(self.rank(model, test_data))
