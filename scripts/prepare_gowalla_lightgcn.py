"""Prepare the LightGCN Gowalla split as a RecBole benchmark dataset.

The source files use one row per user:

    user_id item_id item_id ...

This script merges LightGCN's published train/test interactions, creates a
deterministic project-local train/valid/test split, and writes RecBole atomic
``.inter`` files.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen


SOURCE_URLS = {
    "train.txt": "https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/train.txt",
    "test.txt": "https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/test.txt",
    "user_list.txt": "https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/user_list.txt",
    "item_list.txt": "https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/item_list.txt",
}

INTER_HEADER = "user_id:token\titem_id:token\n"


def download_sources(raw_dir: Path, force: bool = False) -> None:
    """Download LightGCN Gowalla source files if needed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in SOURCE_URLS.items():
        destination = raw_dir / filename
        if destination.exists() and destination.stat().st_size > 0 and not force:
            print(f"Using existing {destination}")
            continue

        print(f"Downloading {url}")
        with urlopen(url) as response, tempfile.NamedTemporaryFile(
            delete=False, dir=raw_dir
        ) as tmp_file:
            shutil.copyfileobj(response, tmp_file)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(destination)


def parse_lightgcn_interactions(path: Path) -> dict[int, list[int]]:
    """Read LightGCN user rows and deduplicate item ids per user."""
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


def merge_user_interactions(
    sources: Iterable[dict[int, list[int]]],
) -> dict[int, list[int]]:
    """Merge multiple user->items mappings while preserving first-seen item order."""
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


def split_user_items(
    item_ids: list[int],
    split: tuple[float, float, float],
    seed: int,
    user_id: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split one user's items into train/valid/test deterministically."""
    if len(item_ids) < 3:
        return list(item_ids), [], []

    shuffled = list(item_ids)
    random.Random(seed + user_id).shuffle(shuffled)

    total = len(shuffled)
    valid_count = int(round(total * split[1]))
    test_count = int(round(total * split[2]))

    valid_count = max(1, valid_count)
    test_count = max(1, test_count)

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

    train_items = shuffled[:train_count]
    valid_items = shuffled[train_count : train_count + valid_count]
    test_items = shuffled[train_count + valid_count :]
    return train_items, valid_items, test_items


def split_interactions(
    interactions: dict[int, list[int]],
    split: tuple[float, float, float],
    seed: int,
) -> dict[str, list[tuple[int, int]]]:
    """Create deterministic internal split rows."""
    rows = {"train": [], "valid": [], "test": []}
    for user_id, item_ids in sorted(interactions.items()):
        train_items, valid_items, test_items = split_user_items(
            item_ids, split=split, seed=seed, user_id=user_id
        )
        rows["train"].extend((user_id, item_id) for item_id in train_items)
        rows["valid"].extend((user_id, item_id) for item_id in valid_items)
        rows["test"].extend((user_id, item_id) for item_id in test_items)
    return rows


def write_recbole_inter(path: Path, rows: Iterable[tuple[int, int]]) -> int:
    """Write RecBole interaction rows and return row count."""
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(INTER_HEADER)
        for user_id, item_id in rows:
            file.write(f"{user_id}\t{item_id}\n")
            count += 1
    return count


def validate_split(rows_by_split: dict[str, list[tuple[int, int]]]) -> None:
    """Validate that generated split rows do not overlap."""
    seen: dict[tuple[int, int], str] = {}
    for split_name, rows in rows_by_split.items():
        for row in rows:
            previous = seen.get(row)
            if previous is not None:
                raise ValueError(
                    f"Interaction {row} appears in both {previous} and {split_name}"
                )
            seen[row] = split_name


def prepare_gowalla(
    output: Path,
    seed: int,
    split: tuple[float, float, float],
    force_download: bool = False,
) -> dict[str, int]:
    """Download, split, validate, and write the Gowalla RecBole dataset."""
    if len(split) != 3:
        raise ValueError("split must contain train, valid, and test ratios")
    if any(ratio < 0 for ratio in split):
        raise ValueError("split ratios must be non-negative")
    if abs(sum(split) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1.0")

    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw-lightgcn"
    download_sources(raw_dir, force=force_download)

    train = parse_lightgcn_interactions(raw_dir / "train.txt")
    test = parse_lightgcn_interactions(raw_dir / "test.txt")
    interactions = merge_user_interactions([train, test])
    rows_by_split = split_interactions(interactions, split=split, seed=seed)
    validate_split(rows_by_split)

    dataset_name = output.name
    counts = {
        "users": len(interactions),
        "items": len({item_id for items in interactions.values() for item_id in items}),
        "source_interactions": sum(len(items) for items in interactions.values()),
    }

    for split_name, rows in rows_by_split.items():
        count = write_recbole_inter(
            output / f"{dataset_name}.{split_name}.inter",
            rows,
        )
        counts[f"{split_name}_interactions"] = count

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LightGCN Gowalla as a RecBole benchmark dataset."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset") / "gowalla-1m",
        help="Dataset output directory.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALID", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Per-user split ratios.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download source files even when they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = prepare_gowalla(
        output=args.output,
        seed=args.seed,
        split=tuple(args.split),
        force_download=args.force_download,
    )

    print("Prepared Gowalla dataset:")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
