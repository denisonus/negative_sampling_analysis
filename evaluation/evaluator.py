"""Evaluation Utilities using RecBole metrics."""

import numpy as np

import torch
from collections import defaultdict
from tqdm import tqdm
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
        user_item_dict,
        num_items,
        metrics=["Recall", "NDCG", "MRR", "Hit"],
        topk=[5, 10, 20],
        device="cpu",
    ):
        self.user_item_dict = user_item_dict
        self.num_items = num_items
        self.metrics = [m.lower() for m in metrics]
        self.topk = topk
        self.max_k = max(topk)
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

        mock_config = MockConfig(topk)
        self.metric_instances = {
            name: self.METRIC_CLASSES[name](mock_config)
            for name in self.metrics
            if name in self.METRIC_CLASSES
        }

    def _extract_interactions(self, test_data):
        """Extract user-item interactions from test data."""
        test_user_items = defaultdict(set)

        for batch in test_data:
            if isinstance(batch, tuple) and len(batch) == 4:
                # RecBole FullSortEvalDataLoader format:
                # (interaction, row_idx, pos_len_list, user_idx)
                interaction, row_idx, pos_len_list, user_idx = batch
                if isinstance(row_idx, tuple) and len(row_idx) == 2:
                    batch_indices, item_ids = row_idx
                    user_ids = user_idx.cpu().numpy()  # actual user IDs
                    batch_indices = batch_indices.cpu().numpy()
                    item_ids = item_ids.cpu().numpy()

                    for batch_idx, item_id in zip(batch_indices, item_ids):
                        if batch_idx < len(user_ids):
                            user_id = user_ids[batch_idx]
                            test_user_items[int(user_id)].add(int(item_id))

            elif isinstance(batch, tuple) and len(batch) == 2:
                users, items = batch
                users = users.cpu().numpy() if torch.is_tensor(users) else users
                items = items.cpu().numpy() if torch.is_tensor(items) else items
                for user, item in zip(users, items):
                    test_user_items[int(user)].add(int(item))

            elif hasattr(batch, "__getitem__"):
                try:
                    users = batch["user_id"]  # type: ignore[index]
                    items = batch["item_id"]  # type: ignore[index]
                    users = users.cpu().numpy() if torch.is_tensor(users) else users
                    items = items.cpu().numpy() if torch.is_tensor(items) else items
                    for user, item in zip(users, items):
                        test_user_items[int(user)].add(int(item))
                except (KeyError, TypeError):
                    pass

        return test_user_items

    def evaluate(self, model, test_data, train_mask=None):
        """Evaluate model on test data."""
        model.eval()
        test_user_items = self._extract_interactions(test_data)

        if not test_user_items:
            return {f"{m}@{k}": 0.0 for m in self.metrics for k in self.topk}

        users_list = list(test_user_items.keys())
        batch_size = 256
        all_pos_index, all_pos_len = [], []

        with torch.no_grad():
            for start_idx in tqdm(
                range(0, len(users_list), batch_size), desc="Evaluating"
            ):
                end_idx = min(start_idx + batch_size, len(users_list))
                batch_users = users_list[start_idx:end_idx]
                user_ids = torch.tensor(
                    batch_users, dtype=torch.long, device=self.device
                )

                scores = model.predict(user_ids)

                # Mask training items
                for i, user in enumerate(batch_users):
                    mask_items = (train_mask or {}).get(
                        user, set()
                    ) | self.user_item_dict.get(user, set())
                    for item in mask_items:
                        if item < scores.size(1):
                            scores[i, item] = float("-inf")

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
