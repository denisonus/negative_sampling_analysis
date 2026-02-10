"""Evaluation Utilities using RecBole metrics."""

import numpy as np

import torch
from collections import defaultdict
from recbole.evaluator.metrics import Hit, Recall, NDCG, MRR, Precision, MAP


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

    def __init__(
        self,
        num_items,
        metrics=["Recall", "NDCG", "MRR", "Hit"],
        topk=[5, 10, 20],
        device="cpu",
    ):
        self.num_items = num_items
        self.metrics = [m.lower() for m in metrics]
        self.topk = topk
        self.max_k = max(topk)
        self.device = device

        mock_config = MockConfig(topk)
        self.metric_instances = {
            name: self.METRIC_CLASSES[name](mock_config)
            for name in self.metrics
            if name in self.METRIC_CLASSES
        }

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

    def evaluate(self, model, test_data):
        """Evaluate model on test data."""
        model.eval()
        test_user_items = self._extract_interactions(test_data)

        if not test_user_items:
            return {f"{m}@{k}": 0.0 for m in self.metrics for k in self.topk}

        uid2history = (
            test_data.uid2history_item if hasattr(test_data, "uid2history_item") else {}
        )

        users_list = list(test_user_items.keys())
        batch_size = 256
        all_pos_index, all_pos_len = [], []

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
                        pos_row = np.array(
                            [item in ground_truth for item in topk_idx[i]]
                        )
                        all_pos_index.append(pos_row)
                        all_pos_len.append(len(ground_truth))

        if not all_pos_index:
            return {f"{m}@{k}": 0.0 for m in self.metrics for k in self.topk}

        pos_index = np.array(all_pos_index)
        pos_len = np.array(all_pos_len)

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
