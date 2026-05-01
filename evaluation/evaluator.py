"""Evaluation Utilities using RecBole metrics."""

import numpy as np

import torch
from collections import defaultdict
from recbole.evaluator.metrics import Hit, Recall, NDCG, MRR, Precision, MAP

if not hasattr(np, "float"):  # pragma: no cover - RecBole still references np.float
    setattr(np, "float", float)


class MockConfig:
    """Mock config object for RecBole metrics."""

    def __init__(self, topk):
        self._config = {"topk": topk, "metric_decimal_place": 4}

    def __getitem__(self, key):
        return self._config.get(key)


class Evaluator:
    """Evaluator for recommendation models using RecBole's standard metrics."""

    METRIC_CLASSES = {
        "hit": Hit,
        "recall": Recall,
        "ndcg": NDCG,
        "mrr": MRR,
        "precision": Precision,
        "map": MAP,
    }
    ACTIVITY_BUCKET_ORDER = ("0", "1-5", "6-20", "21+")

    def __init__(
        self,
        num_items,
        metrics=["Recall", "NDCG", "MRR", "Hit"],
        topk=[5, 10, 20],
        device="cpu",
        batch_size=256,
    ):
        self.num_items = num_items
        self.metrics = [m.lower() for m in metrics]
        self.topk = topk
        self.max_k = max(topk)
        self.device = device
        self.batch_size = max(int(batch_size), 1)

        mock_config = MockConfig(topk)
        self.metric_instances = {
            name: self.METRIC_CLASSES[name](mock_config)
            for name in self.metrics
            if name in self.METRIC_CLASSES
        }

    def _empty_results(self):
        return {f"{m}@{k}": 0.0 for m in self.metrics for k in self.topk}

    def _empty_rankings(self):
        return {
            "user_ids": np.array([], dtype=np.int64),
            "topk_items": np.empty((0, self.max_k), dtype=np.int64),
            "pos_index": np.empty((0, self.max_k), dtype=bool),
            "pos_len": np.array([], dtype=np.int64),
        }

    def _get_metric_instance(self, metric_name):
        metric_name = metric_name.lower()
        if metric_name in self.metric_instances:
            return self.metric_instances[metric_name]
        return self.METRIC_CLASSES[metric_name](MockConfig(self.topk))

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

        metric_instance = self._get_metric_instance(metric_name)
        if metric_name in ["recall", "ndcg", "map"]:
            metric_values = metric_instance.metric_info(pos_index, pos_len)
        else:
            metric_values = metric_instance.metric_info(pos_index)

        return np.asarray(metric_values, dtype=np.float64)

    def _extract_interactions(self, test_data):
        """Extract user-item interactions from test data."""
        test_user_items = defaultdict(set)
        uid2pos = test_data.uid2positive_item
        uid_list = test_data.uid_list.numpy()

        for uid in uid_list:
            pos_items = uid2pos[uid]
            if hasattr(pos_items, "tolist"):
                test_user_items[int(uid)] = set(pos_items.tolist())
            elif hasattr(pos_items, "__iter__"):
                test_user_items[int(uid)] = set(int(i) for i in pos_items)
            else:
                test_user_items[int(uid)] = {int(pos_items)}

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
        for metric_name, metric_instance in self.metric_instances.items():
            if metric_name in ["recall", "ndcg", "map"]:
                metric_values = metric_instance.metric_info(pos_index, pos_len)
            else:
                metric_values = metric_instance.metric_info(pos_index)

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
