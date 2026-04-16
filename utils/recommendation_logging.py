"""Recommendation logging utilities with human-readable item metadata."""

import csv
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

DEFAULT_RECOMMENDATION_LOG_USERS = 10
DEFAULT_RECOMMENDATION_LOG_TOPK = 10


def _normalize_field_name(name: str) -> str:
    return str(name).split(":", 1)[0]


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_sequence(values: Any) -> list:
    values = _to_python_scalar(values)
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


def _stringify(value: Any) -> Optional[str]:
    value = _to_python_scalar(value)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _sequence_to_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bytes):
        text = value.decode("utf-8").strip()
        return text or None

    tokens = []
    for token in _to_sequence(value):
        text = _stringify(token)
        if text is None or text in {"", "0", "[PAD]"}:
            continue
        tokens.append(text)

    if not tokens:
        return None
    return " ".join(tokens)


def _genres_from_value(value: Any) -> list[str]:
    text = _sequence_to_text(value)
    if text is None:
        return []
    return [genre for genre in text.split() if genre]


def _values_from_column(column: Any) -> list:
    if column is None:
        return []
    if isinstance(column, torch.Tensor):
        return column.detach().cpu().tolist()
    if isinstance(column, np.ndarray):
        return column.tolist()
    if hasattr(column, "tolist"):
        values = column.tolist()
        if isinstance(values, list):
            return values
        return [values]
    return list(column)


def _maybe_get_column(table: Any, field_name: str) -> Optional[list]:
    if table is None:
        return None

    columns = getattr(table, "columns", None)
    if columns is None or field_name not in columns:
        return None

    return _values_from_column(table[field_name])


def _read_item_metadata_file(item_file: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(item_file):
        return {}

    with open(item_file, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        if not reader.fieldnames:
            return {}

        normalized_fields = {
            field_name: _normalize_field_name(field_name)
            for field_name in reader.fieldnames
        }
        item_field = next(
            (
                field_name
                for field_name, normalized in normalized_fields.items()
                if normalized == "item_id"
            ),
            reader.fieldnames[0],
        )
        title_field = next(
            (
                field_name
                for field_name, normalized in normalized_fields.items()
                if normalized in {"movie_title", "title"}
            ),
            None,
        )
        genre_field = next(
            (
                field_name
                for field_name, normalized in normalized_fields.items()
                if normalized in {"genre", "genres"}
            ),
            None,
        )

        metadata = {}
        source_name = os.path.basename(item_file)
        for row in reader:
            raw_item_id = row.get(item_field)
            if raw_item_id is None:
                continue

            metadata[str(raw_item_id)] = {
                "raw_item_id": str(raw_item_id),
                "title": _sequence_to_text(row.get(title_field)),
                "genres": _genres_from_value(row.get(genre_field)),
                "metadata_source": source_name,
            }

    return metadata


def _item_file_path(data_path: str, dataset_name: Optional[str]) -> Optional[str]:
    if not dataset_name:
        return None
    return os.path.join(data_path, dataset_name, f"{dataset_name}.item")


def _extract_internal_item_tokens(dataset: Any) -> Optional[list[str]]:
    iid_field = getattr(dataset, "iid_field", None)
    if iid_field is None:
        return None

    field2id_token = getattr(dataset, "field2id_token", None)
    if isinstance(field2id_token, dict) and iid_field in field2id_token:
        return [
            _stringify(token)
            for token in _to_sequence(field2id_token[iid_field])
        ]

    id2token = getattr(dataset, "id2token", None)
    if callable(id2token):
        num_items = int(dataset.num(iid_field))
        return [
            _stringify(token)
            for token in _to_sequence(id2token(iid_field, np.arange(num_items)))
        ]

    return None


class ItemMetadataLookup:
    """Resolve internal item IDs to readable metadata."""

    def __init__(
        self,
        item_metadata: Dict[int, Dict[str, Any]],
        sources: Dict[str, Any],
    ):
        self.item_metadata = item_metadata
        self.sources = sources

    @classmethod
    def build(
        cls,
        dataset: Any,
        dataset_name: str,
        data_path: str = "dataset/",
        fallback_dataset_name: Optional[str] = "ml-1m",
    ) -> "ItemMetadataLookup":
        num_items = int(dataset.num(dataset.iid_field))
        item_metadata: Dict[int, Dict[str, Any]] = {}
        direct_coverage = 0

        item_feat = getattr(dataset, "item_feat", None)
        item_ids = _maybe_get_column(item_feat, "item_id")
        titles = _maybe_get_column(item_feat, "movie_title") or _maybe_get_column(
            item_feat, "title"
        )
        genres = _maybe_get_column(item_feat, "genre") or _maybe_get_column(
            item_feat, "genres"
        )
        if item_ids is not None and (titles is not None or genres is not None):
            for idx, internal_id in enumerate(item_ids):
                item_metadata[int(internal_id)] = {
                    "raw_item_id": _stringify(internal_id),
                    "title": (
                        _sequence_to_text(titles[idx]) if titles is not None else None
                    ),
                    "genres": (
                        _genres_from_value(genres[idx]) if genres is not None else []
                    ),
                    "metadata_source": "dataset.item_feat",
                }
            direct_coverage = len(item_metadata)

        primary_file = _item_file_path(data_path, dataset_name)
        fallback_file = _item_file_path(data_path, fallback_dataset_name)
        primary_raw_metadata = (
            _read_item_metadata_file(primary_file) if primary_file is not None else {}
        )
        fallback_raw_metadata = {}
        if fallback_file is not None and fallback_file != primary_file:
            fallback_raw_metadata = _read_item_metadata_file(fallback_file)

        internal_tokens = _extract_internal_item_tokens(dataset)
        primary_matches = 0
        fallback_matches = 0

        for internal_id in range(num_items):
            existing = item_metadata.get(internal_id)
            if existing is not None and existing.get("title"):
                continue

            raw_item_id = None
            if internal_tokens is not None and internal_id < len(internal_tokens):
                raw_item_id = internal_tokens[internal_id]
            if raw_item_id is None:
                raw_item_id = str(internal_id)

            metadata = primary_raw_metadata.get(raw_item_id)
            if metadata is not None:
                primary_matches += 1
            else:
                metadata = fallback_raw_metadata.get(raw_item_id)
                if metadata is not None:
                    fallback_matches += 1

            if metadata is None:
                item_metadata.setdefault(
                    internal_id,
                    {
                        "raw_item_id": raw_item_id,
                        "title": None,
                        "genres": [],
                        "metadata_source": "unavailable",
                    },
                )
                continue

            merged = dict(metadata)
            merged["raw_item_id"] = raw_item_id
            item_metadata[internal_id] = merged

        sources = {
            "dataset": dataset_name,
            "data_path": data_path,
            "primary_item_file": (
                primary_file
                if primary_file and os.path.exists(primary_file)
                else None
            ),
            "fallback_item_file": (
                fallback_file
                if fallback_file and os.path.exists(fallback_file)
                else None
            ),
            "direct_feature_rows": direct_coverage,
            "primary_matches": primary_matches,
            "fallback_matches": fallback_matches,
            "covered_items": sum(
                1 for metadata in item_metadata.values() if metadata.get("title")
            ),
            "num_items": num_items,
        }
        return cls(item_metadata=item_metadata, sources=sources)

    def describe_item(self, item_id: Any) -> Dict[str, Any]:
        internal_item_id = int(item_id)
        metadata = self.item_metadata.get(
            internal_item_id,
            {
                "raw_item_id": str(internal_item_id),
                "title": None,
                "genres": [],
                "metadata_source": "unavailable",
            },
        )
        return {
            "internal_item_id": internal_item_id,
            "raw_item_id": metadata.get("raw_item_id"),
            "title": metadata.get("title"),
            "genres": list(metadata.get("genres", [])),
            "metadata_source": metadata.get("metadata_source", "unavailable"),
        }


def extract_test_user_items(test_data) -> Dict[int, set[int]]:
    """Extract held-out positive items from RecBole evaluation data."""
    test_user_items = {}
    uid2pos = test_data.uid2positive_item
    uid_list = _to_sequence(test_data.uid_list)

    for uid in uid_list:
        pos_items = uid2pos[uid]
        if hasattr(pos_items, "tolist"):
            test_user_items[int(uid)] = {int(item) for item in pos_items.tolist()}
        elif hasattr(pos_items, "__iter__"):
            test_user_items[int(uid)] = {int(item) for item in pos_items}
        else:
            test_user_items[int(uid)] = {int(pos_items)}

    return test_user_items


def build_recommendation_log(
    rankings: dict,
    test_user_items: Dict[int, set[int]],
    item_metadata: ItemMetadataLookup,
    topk: int = DEFAULT_RECOMMENDATION_LOG_TOPK,
    num_users: int = DEFAULT_RECOMMENDATION_LOG_USERS,
) -> dict:
    """Build a compact, deterministic recommendation log from final rankings."""
    user_ids = _to_sequence(rankings.get("user_ids"))
    topk_items = _to_python_scalar(rankings.get("topk_items"))
    pos_index = _to_python_scalar(rankings.get("pos_index"))

    selected_topk = max(1, int(topk))
    selected_num_users = max(1, int(num_users))

    if not user_ids or topk_items is None or pos_index is None:
        return {
            "selection": "sorted_user_ids_first_n",
            "requested_user_count": selected_num_users,
            "logged_user_count": 0,
            "topk": selected_topk,
            "metadata_sources": item_metadata.sources,
            "records": [],
        }

    row_by_user = {int(user_id): idx for idx, user_id in enumerate(user_ids)}
    selected_user_ids = sorted(row_by_user.keys())[:selected_num_users]

    records = []
    for user_id in selected_user_ids:
        row_idx = row_by_user[user_id]
        recommended_items = _to_sequence(topk_items[row_idx])[:selected_topk]
        hit_flags = [
            bool(flag) for flag in _to_sequence(pos_index[row_idx])[:selected_topk]
        ]
        if len(hit_flags) < len(recommended_items):
            hit_flags.extend([False] * (len(recommended_items) - len(hit_flags)))

        top_recommendations = []
        for rank, (item_id, is_hit) in enumerate(
            zip(recommended_items, hit_flags), start=1
        ):
            recommendation = item_metadata.describe_item(item_id)
            recommendation["rank"] = rank
            recommendation["is_hit"] = bool(is_hit)
            top_recommendations.append(recommendation)

        ground_truth_items = [
            item_metadata.describe_item(item_id)
            for item_id in sorted(test_user_items.get(user_id, set()))
        ]
        records.append(
            {
                "user_id": int(user_id),
                "num_test_positives": len(ground_truth_items),
                "ground_truth_items": ground_truth_items,
                "top_recommendations": top_recommendations,
                "num_hits_at_logged_k": int(sum(hit_flags)),
            }
        )

    return {
        "selection": "sorted_user_ids_first_n",
        "requested_user_count": selected_num_users,
        "logged_user_count": len(records),
        "topk": selected_topk,
        "metadata_sources": item_metadata.sources,
        "records": records,
    }


def _join_titles(items: list[dict]) -> str:
    titles = [item.get("title") or f"item:{item['internal_item_id']}" for item in items]
    return " | ".join(titles)


def _format_recommendation(rec: dict) -> str:
    title = rec.get("title") or f"item:{rec['internal_item_id']}"
    genres = ", ".join(rec.get("genres", []))
    suffix = "hit" if rec.get("is_hit") else "miss"
    if genres:
        return f"{rec['rank']}. {title} [{genres}] ({suffix})"
    return f"{rec['rank']}. {title} ({suffix})"


def build_recommendation_summary_rows(recommendation_log: dict) -> list[dict]:
    """Return one compact row per user for CSV export."""
    rows = []
    for record in recommendation_log.get("records", []):
        recommendations = record.get("top_recommendations", [])
        hit_titles = [rec for rec in recommendations if rec.get("is_hit")]
        rows.append(
            {
                "user_id": record["user_id"],
                "num_test_positives": record["num_test_positives"],
                "num_hits_at_logged_k": record["num_hits_at_logged_k"],
                "ground_truth_titles": _join_titles(record.get("ground_truth_items", [])),
                "hit_titles": _join_titles(hit_titles),
                "top_recommendations": " | ".join(
                    _format_recommendation(rec) for rec in recommendations
                ),
            }
        )
    return rows


def build_recommendation_detail_rows(recommendation_log: dict) -> list[dict]:
    """Return one row per recommendation for CSV export."""
    rows = []
    for record in recommendation_log.get("records", []):
        ground_truth_titles = _join_titles(record.get("ground_truth_items", []))
        for recommendation in record.get("top_recommendations", []):
            rows.append(
                {
                    "user_id": record["user_id"],
                    "num_test_positives": record["num_test_positives"],
                    "num_hits_at_logged_k": record["num_hits_at_logged_k"],
                    "ground_truth_titles": ground_truth_titles,
                    "rank": recommendation["rank"],
                    "is_hit": recommendation["is_hit"],
                    "recommended_title": recommendation.get("title"),
                    "recommended_genres": " | ".join(
                        recommendation.get("genres", [])
                    ),
                    "internal_item_id": recommendation["internal_item_id"],
                    "raw_item_id": recommendation.get("raw_item_id"),
                }
            )
    return rows


def write_csv_rows(path: str, rows: list[dict]) -> None:
    """Write flat rows to CSV, creating an empty header-only file when needed."""
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        if not fieldnames:
            return
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
