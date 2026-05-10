"""Dataset loading utilities for the experiment runner."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

USER_ID_FIELD = "user_id"
ITEM_ID_FIELD = "item_id"
SPLIT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class SimpleDataset:
    """Minimal dataset metadata used by the model and runner."""

    dataset_name: str
    num_users: int
    num_items: int
    user_index: dict[int, int]
    item_index: dict[int, int]
    uid_field: str = USER_ID_FIELD
    iid_field: str = ITEM_ID_FIELD

    def num(self, field: str) -> int:
        if field == self.uid_field:
            return self.num_users
        if field == self.iid_field:
            return self.num_items
        raise KeyError(f"Unknown dataset field: {field}")


@dataclass(frozen=True)
class EvalSplit:
    """Full-sort evaluation data for one split."""

    uid_list: torch.Tensor
    uid2positive_item: np.ndarray
    uid2history_item: np.ndarray


@dataclass(frozen=True)
class ParsedInteractions:
    """Internal positive interactions before train/validation/test splitting."""

    rows: list[tuple[int, int, int]]
    dataset: SimpleDataset


def load_dataset(
    dataset_name,
    data_path="dataset/",
    min_rating=4,
    seed=42,
):
    """Load a supported dataset into local train and evaluation structures."""
    dataset_dir = Path(data_path) / dataset_name
    if dataset_name in {"ml-100k", "ml-1m"}:
        parsed = _load_movielens(dataset_name, dataset_dir, min_rating=min_rating)
        train, valid, test = _split_timestamped_interactions(parsed.rows)
        return (
            parsed.dataset,
            train,
            _build_eval_split(valid, history_interactions=train, num_users=parsed.dataset.num_users),
            _build_eval_split(
                test,
                history_interactions=[*train, *valid],
                num_users=parsed.dataset.num_users,
            ),
        )

    if dataset_name == "gowalla-1m":
        dataset, train, valid, test = _load_gowalla(dataset_dir, seed=seed)
        return (
            dataset,
            train,
            _build_eval_split(valid, history_interactions=train, num_users=dataset.num_users),
            _build_eval_split(
                test,
                history_interactions=[*train, *valid],
                num_users=dataset.num_users,
            ),
        )

    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file is missing: {path}")
    return path


def _read_ml100k_ratings(dataset_dir: Path):
    path = _require_file(dataset_dir / "u.data")
    with path.open("r", encoding="latin-1") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"Invalid ml-100k rating row at {path}:{line_number}")
            user_id, item_id, rating, timestamp = parts
            yield int(user_id), int(item_id), float(rating), int(timestamp)


def _read_ml1m_ratings(dataset_dir: Path):
    path = _require_file(dataset_dir / "ratings.dat")
    with path.open("r", encoding="latin-1") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 4:
                raise ValueError(f"Invalid ml-1m rating row at {path}:{line_number}")
            user_id, item_id, rating, timestamp = parts
            yield int(user_id), int(item_id), float(rating), int(timestamp)


def _load_movielens(dataset_name: str, dataset_dir: Path, min_rating) -> ParsedInteractions:
    if min_rating is None:
        min_rating = 0

    reader = _read_ml100k_ratings if dataset_name == "ml-100k" else _read_ml1m_ratings
    raw_rows = [
        (user_id, item_id, timestamp)
        for user_id, item_id, rating, timestamp in reader(dataset_dir)
        if rating >= min_rating
    ]
    if not raw_rows:
        raise ValueError(
            f"Dataset '{dataset_name}' has no interactions after applying min_rating={min_rating}"
        )

    user_index = {
        user_id: idx for idx, user_id in enumerate(sorted({row[0] for row in raw_rows}))
    }
    item_index = {
        item_id: idx for idx, item_id in enumerate(sorted({row[1] for row in raw_rows}))
    }
    rows = [
        (user_index[user_id], item_index[item_id], timestamp)
        for user_id, item_id, timestamp in raw_rows
    ]

    dataset = SimpleDataset(
        dataset_name=dataset_name,
        num_users=len(user_index),
        num_items=len(item_index),
        user_index=user_index,
        item_index=item_index,
    )
    return ParsedInteractions(rows=rows, dataset=dataset)


def _split_counts(total: int, ratios=SPLIT_RATIOS) -> list[int]:
    normalized = [ratio / sum(ratios) for ratio in ratios]
    counts = [int(ratio * total) for ratio in normalized]
    counts[0] = total - sum(counts[1:])

    for index in range(1, len(normalized)):
        if counts[0] <= 1:
            break
        ratio = normalized[-index]
        if 0 < ratio * total < 1:
            counts[-index] += 1
            counts[0] -= 1

    return counts


def _split_timestamped_interactions(
    rows: Iterable[tuple[int, int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    rows_by_user = defaultdict(list)
    for user_id, item_id, timestamp in rows:
        rows_by_user[int(user_id)].append((int(item_id), int(timestamp)))

    splits = ([], [], [])
    for user_id in sorted(rows_by_user):
        user_rows = sorted(rows_by_user[user_id], key=lambda row: row[1])
        counts = _split_counts(len(user_rows))
        start = 0
        for split_rows, count in zip(splits, counts):
            end = start + count
            split_rows.extend((user_id, item_id) for item_id, _ in user_rows[start:end])
            start = end

    return splits


def _parse_lightgcn_interactions(path: Path) -> dict[int, list[int]]:
    _require_file(path)
    interactions: dict[int, list[int]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                user_id = int(parts[0])
                item_ids = [int(item_id) for item_id in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid integer in {path}:{line_number}") from exc
            interactions[user_id] = list(dict.fromkeys(item_ids))
    return interactions


def _merge_user_interactions(
    sources: Iterable[dict[int, list[int]]],
) -> dict[int, list[int]]:
    merged = defaultdict(list)
    seen_by_user: dict[int, set[int]] = defaultdict(set)
    for source in sources:
        for user_id, item_ids in source.items():
            seen = seen_by_user[user_id]
            for item_id in item_ids:
                if item_id in seen:
                    continue
                merged[user_id].append(item_id)
                seen.add(item_id)
    return dict(sorted(merged.items()))


def _split_gowalla_user_items(
    item_ids: list[int],
    seed: int,
    user_id: int,
    split=SPLIT_RATIOS,
) -> tuple[list[int], list[int], list[int]]:
    if len(item_ids) < 3:
        return list(item_ids), [], []

    shuffled = list(item_ids)
    import random

    random.Random(seed + user_id).shuffle(shuffled)
    total = len(shuffled)
    valid_count = max(1, int(round(total * split[1])))
    test_count = max(1, int(round(total * split[2])))

    if valid_count + test_count >= total:
        overflow = valid_count + test_count - (total - 1)
        reduce_valid = min(valid_count, overflow)
        valid_count -= reduce_valid
        overflow -= reduce_valid
        test_count = max(1, test_count - overflow)

    train_count = total - valid_count - test_count
    if train_count < 1:
        train_count = 1
        available = total - train_count
        test_count = min(test_count, available)
        valid_count = max(0, available - test_count)

    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + valid_count],
        shuffled[train_count + valid_count :],
    )


def _load_gowalla(
    dataset_dir: Path,
    seed: int,
) -> tuple[SimpleDataset, list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    raw_dir = dataset_dir / "raw-lightgcn"
    train_source = _parse_lightgcn_interactions(raw_dir / "train.txt")
    test_source = _parse_lightgcn_interactions(raw_dir / "test.txt")
    merged = _merge_user_interactions([train_source, test_source])
    if not merged:
        raise ValueError(f"Dataset '{dataset_dir.name}' has no raw interactions")

    user_index = {user_id: idx for idx, user_id in enumerate(sorted(merged))}
    all_item_ids = sorted({item_id for item_ids in merged.values() for item_id in item_ids})
    item_index = {item_id: idx for idx, item_id in enumerate(all_item_ids)}
    splits = ([], [], [])

    for external_user_id, item_ids in merged.items():
        user_id = user_index[external_user_id]
        indexed_items = [item_index[item_id] for item_id in item_ids]
        user_splits = _split_gowalla_user_items(
            indexed_items,
            seed=seed,
            user_id=user_id,
        )
        for split_rows, split_items in zip(splits, user_splits):
            split_rows.extend((user_id, item_id) for item_id in split_items)

    dataset = SimpleDataset(
        dataset_name=dataset_dir.name,
        num_users=len(user_index),
        num_items=len(item_index),
        user_index=user_index,
        item_index=item_index,
    )
    return dataset, splits[0], splits[1], splits[2]


def _build_eval_split(
    positive_interactions: Iterable[tuple[int, int]],
    history_interactions: Iterable[tuple[int, int]],
    num_users: int,
) -> EvalSplit:
    positives = [set() for _ in range(num_users)]
    history = [set() for _ in range(num_users)]

    for user_id, item_id in history_interactions:
        history[int(user_id)].add(int(item_id))
    for user_id, item_id in positive_interactions:
        positives[int(user_id)].add(int(item_id))

    uid_list = [user_id for user_id, items in enumerate(positives) if items]
    uid2positive_item = np.empty(num_users, dtype=object)
    uid2history_item = np.empty(num_users, dtype=object)
    for user_id in range(num_users):
        uid2positive_item[user_id] = np.asarray(sorted(positives[user_id]), dtype=np.int64)
        uid2history_item[user_id] = np.asarray(sorted(history[user_id]), dtype=np.int64)

    return EvalSplit(
        uid_list=torch.tensor(uid_list, dtype=torch.long),
        uid2positive_item=uid2positive_item,
        uid2history_item=uid2history_item,
    )


def build_user_item_dict_from_train(train_interactions):
    """Build user-item interaction dictionary from training interactions only."""
    user_item_dict = defaultdict(set)
    for user, item in train_interactions:
        user_item_dict[int(user)].add(int(item))
    return dict(user_item_dict)


def compute_item_popularity_from_train(train_interactions, num_items):
    """Compute item popularity from training interactions only."""
    popularity = np.zeros(num_items)
    for _, item in train_interactions:
        popularity[int(item)] += 1
    return popularity + 1


def compute_user_interaction_counts_from_train(train_interactions):
    """Count training interactions per user."""
    counts = defaultdict(int)
    for user, _ in train_interactions:
        counts[int(user)] += 1
    return dict(counts)


def get_train_interactions(train_data):
    """Return user-item pairs from the local training split."""
    return [(int(user), int(item)) for user, item in train_data]


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
