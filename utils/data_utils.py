"""Data Utilities for RecBole Integration."""

import torch
import numpy as np
from collections import defaultdict
from recbole.config import Config
from recbole.data import create_dataset, data_preparation


def load_recbole_dataset(dataset_name, data_path="dataset/"):
    """Load dataset using RecBole."""
    config_dict = {
        "data_path": data_path,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "metrics": ["Recall", "NDCG", "MRR", "Hit"],
        "topk": [5, 10, 20],
        "valid_metric": "NDCG@10",
    }

    config = Config(model="BPR", dataset=dataset_name, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return config, dataset, train_data, valid_data, test_data


def build_user_item_dict(dataset):
    """Build user-item interaction dictionary."""
    user_item_dict = defaultdict(set)

    inter_feat = dataset.inter_feat
    users = inter_feat[dataset.uid_field].numpy()
    items = inter_feat[dataset.iid_field].numpy()

    for user, item in zip(users, items):
        user_item_dict[user].add(item)

    return dict(user_item_dict)


def compute_item_popularity(dataset):
    """Compute item popularity (frequency counts)."""
    items = dataset.inter_feat[dataset.iid_field].numpy()
    num_items = dataset.num(dataset.iid_field)

    popularity = np.zeros(num_items)
    for item in items:
        popularity[item] += 1

    return popularity + 1  # Add smoothing


def get_train_interactions(train_data):
    """Extract user-item pairs from training data."""
    interactions = []
    for batch_data in train_data:
        user_ids = batch_data["user_id"].numpy()
        item_ids = batch_data["item_id"].numpy()
        for user, item in zip(user_ids, item_ids):
            interactions.append((user, item))
    return interactions


class SimpleDataLoader:
    """Simple dataloader for training two-tower model."""

    def __init__(self, interactions, batch_size, shuffle=True):
        self.interactions = interactions
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(interactions)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            users, items = [], []
            for idx in batch_indices:
                user, item = self.interactions[idx]
                users.append(user)
                items.append(item)

            yield (
                torch.tensor(users, dtype=torch.long),
                torch.tensor(items, dtype=torch.long),
            )

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
