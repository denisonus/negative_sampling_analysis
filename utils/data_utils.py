"""Data Utilities for RecBole Integration."""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

try:
    from recbole.utils.enum_type import FeatureType
except ImportError:  # pragma: no cover - fallback for RecBole API differences
    FeatureType = None


FEATURE_PROFILES = {
    "ml-100k": {
        "user": ["age", "gender", "occupation"],
        "item": ["release_year"],
    }
}


def get_feature_profile(dataset_name):
    """Return the configured side-feature profile for a dataset, if any."""
    profile = FEATURE_PROFILES.get(dataset_name)
    return deepcopy(profile) if profile is not None else None


def _build_load_columns(dataset_name, feature_aware=False):
    load_col = {"inter": ["user_id", "item_id", "rating", "timestamp"]}
    if not feature_aware:
        return load_col

    profile = get_feature_profile(dataset_name)
    if profile is None:
        raise ValueError(
            f"Feature-aware mode is not configured for dataset '{dataset_name}'"
        )

    if profile.get("user"):
        load_col["user"] = ["user_id", *profile["user"]]
    if profile.get("item"):
        load_col["item"] = ["item_id", *profile["item"]]
    return load_col


def load_recbole_dataset(
    dataset_name, data_path="dataset/", min_rating=4, feature_aware=False
):
    """Load dataset using RecBole."""
    config_dict = {
        "data_path": data_path,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": _build_load_columns(dataset_name, feature_aware=feature_aware),
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "TO",
            "mode": "full",
        },
        "metrics": ["Recall", "NDCG", "MRR", "Hit"],
        "topk": [5, 10, 20],
        "valid_metric": "NDCG@10",
        "val_interval": {"rating": f"[{min_rating},inf)"},
    }

    config = Config(model="BPR", dataset=dataset_name, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return config, dataset, train_data, valid_data, test_data


def _normalize_feature_type(feature_type):
    if FeatureType is not None:
        if feature_type == FeatureType.TOKEN:
            return "token"
        if feature_type == FeatureType.TOKEN_SEQ:
            return "token_seq"

    type_name = getattr(feature_type, "name", str(feature_type)).lower()
    if "token_seq" in type_name:
        return "token_seq"
    if "token" in type_name:
        return "token"
    raise ValueError(f"Unsupported feature type: {feature_type}")


def _align_token_feature(values, entity_ids, entity_count):
    aligned = np.zeros(entity_count, dtype=np.int64)
    aligned[entity_ids] = np.asarray(values, dtype=np.int64)
    return torch.from_numpy(aligned)


def _align_token_seq_feature(values, entity_ids, entity_count):
    sequences = [np.asarray(value, dtype=np.int64) for value in values]
    max_len = max((len(seq) for seq in sequences), default=0)
    aligned = np.zeros((entity_count, max_len), dtype=np.int64)

    for entity_id, seq in zip(entity_ids, sequences):
        if seq.size > 0:
            aligned[int(entity_id), : seq.size] = seq

    return torch.from_numpy(aligned), max_len


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    return np.asarray(values)


def _to_sequence_values(values):
    if isinstance(values, torch.Tensor):
        return [row for row in values.detach().cpu().numpy()]
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _extract_entity_features(dataset, entity_name, id_field, entity_count, field_names):
    feat_table = dataset.user_feat if entity_name == "user" else dataset.item_feat
    if feat_table is None:
        raise ValueError(
            f"Feature-aware mode requested but RecBole did not load {entity_name} features"
        )

    missing_fields = [field for field in field_names if field not in feat_table.columns]
    if missing_fields:
        raise ValueError(
            f"Missing {entity_name} side features for dataset '{dataset.dataset_name}': "
            f"{missing_fields}"
        )

    entity_ids = np.asarray(_to_numpy(feat_table[id_field]), dtype=np.int64)
    if entity_ids.size != entity_count:
        raise ValueError(
            f"Expected {entity_count} {entity_name} rows, found {entity_ids.size}"
        )
    if np.unique(entity_ids).size != entity_count:
        raise ValueError(f"{entity_name} feature IDs are not unique")

    schema = []
    tensors = {}
    for field in field_names:
        feature_type = _normalize_feature_type(dataset.field2type[field])
        values = _to_sequence_values(feat_table[field])
        if feature_type == "token":
            tensors[field] = _align_token_feature(values, entity_ids, entity_count)
            schema.append(
                {
                    "name": field,
                    "type": feature_type,
                    "num_embeddings": int(dataset.num(field)),
                }
            )
            continue

        if feature_type == "token_seq":
            tensor, max_length = _align_token_seq_feature(
                values, entity_ids, entity_count
            )
            tensors[field] = tensor
            schema.append(
                {
                    "name": field,
                    "type": feature_type,
                    "num_embeddings": int(dataset.num(field)),
                    "max_length": max_length,
                }
            )
            continue

        raise ValueError(
            f"Unsupported {entity_name} feature type for field '{field}': {feature_type}"
        )

    return {"schema": schema, "tensors": tensors}


def extract_feature_data(dataset, dataset_name):
    """Extract ID-aligned side-feature tensors for the configured dataset profile."""
    profile = get_feature_profile(dataset_name)
    if profile is None:
        raise ValueError(
            f"Feature-aware mode is not configured for dataset '{dataset_name}'"
        )

    return {
        "dataset": dataset_name,
        "profile": profile,
        "user": _extract_entity_features(
            dataset,
            entity_name="user",
            id_field=dataset.uid_field,
            entity_count=dataset.num(dataset.uid_field),
            field_names=profile.get("user", []),
        ),
        "item": _extract_entity_features(
            dataset,
            entity_name="item",
            id_field=dataset.iid_field,
            entity_count=dataset.num(dataset.iid_field),
            field_names=profile.get("item", []),
        ),
    }


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


def build_user_item_dict_from_train(train_interactions):
    """Build user-item interaction dictionary from training interactions only."""
    user_item_dict = defaultdict(set)
    for user, item in train_interactions:
        user_item_dict[user].add(item)
    return dict(user_item_dict)


def compute_item_popularity_from_train(train_interactions, num_items):
    """Compute item popularity from training interactions only."""
    popularity = np.zeros(num_items)
    for _, item in train_interactions:
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
