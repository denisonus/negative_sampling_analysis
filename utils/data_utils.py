"""Data Utilities for RecBole Integration."""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from utils.experiment_config import COMMON_DEFAULTS

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


def _build_load_columns(dataset_name, feature_aware=False, implicit_feedback=False):
    inter_columns = ["user_id", "item_id"]
    if not implicit_feedback:
        inter_columns.extend(["rating", "timestamp"])
    load_col = {"inter": inter_columns}
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
    dataset_name,
    data_path="dataset/",
    min_rating=4,
    feature_aware=False,
    implicit_feedback=False,
    benchmark_filename=None,
    metrics=None,
    topk=None,
    valid_metric=None,
):
    """Load dataset using RecBole."""
    config_dict = build_recbole_config_dict(
        dataset_name=dataset_name,
        data_path=data_path,
        min_rating=min_rating,
        feature_aware=feature_aware,
        implicit_feedback=implicit_feedback,
        benchmark_filename=benchmark_filename,
        metrics=metrics,
        topk=topk,
        valid_metric=valid_metric,
    )

    config = Config(model="BPR", dataset=dataset_name, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    return config, dataset, train_data, valid_data, test_data


def build_recbole_config_dict(
    dataset_name,
    data_path="dataset/",
    min_rating=4,
    feature_aware=False,
    implicit_feedback=False,
    benchmark_filename=None,
    metrics=None,
    topk=None,
    valid_metric=None,
):
    """Build RecBole config dict for explicit and implicit-feedback datasets."""
    if metrics is None:
        metrics = COMMON_DEFAULTS["metrics"]
    if topk is None:
        topk = COMMON_DEFAULTS["topk"]
    if valid_metric is None:
        valid_metric = COMMON_DEFAULTS["valid_metric"]

    config_dict = {
        "data_path": data_path,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "load_col": _build_load_columns(
            dataset_name,
            feature_aware=feature_aware,
            implicit_feedback=implicit_feedback,
        ),
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO" if implicit_feedback else "TO",
            "mode": "full",
        },
        "metrics": list(metrics),
        "topk": list(topk),
        "valid_metric": valid_metric,
    }

    if benchmark_filename is not None:
        config_dict["benchmark_filename"] = list(benchmark_filename)

    if min_rating is not None and not implicit_feedback:
        config_dict["val_interval"] = {"rating": f"[{min_rating},inf)"}

    return config_dict


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


def compute_user_interaction_counts_from_train(train_interactions):
    """Count training interactions per user."""
    counts = defaultdict(int)
    for user, _ in train_interactions:
        counts[int(user)] += 1
    return dict(counts)


def get_train_interactions(train_data):
    """Extract user-item pairs from training data."""
    interactions = []
    for batch_data in train_data:
        user_ids = batch_data["user_id"].numpy()
        item_ids = batch_data["item_id"].numpy()
        for user, item in zip(user_ids, item_ids):
            interactions.append((user, item))
    return interactions


class TrainLoader:
    """Small training mini-batch loader for user-item pairs."""

    def __init__(self, interactions, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        interactions_array = np.asarray(interactions, dtype=np.int64)
        if interactions_array.size == 0:
            interactions_array = interactions_array.reshape(0, 2)
        if interactions_array.ndim != 2 or interactions_array.shape[1] != 2:
            raise ValueError("TrainLoader interactions must be user-item pairs")

        self.users = interactions_array[:, 0]
        self.items = interactions_array[:, 1]
        self.num_samples = len(interactions_array)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            yield (
                torch.from_numpy(self.users[batch_indices]).long(),
                torch.from_numpy(self.items[batch_indices]).long(),
            )

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
