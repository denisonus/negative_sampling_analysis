"""Lightweight, dataset-agnostic recommendation logging utilities."""

import csv
from typing import Any, Dict, Optional

import numpy as np
import torch

DEFAULT_RECOMMENDATION_LOG_USERS = 10
DEFAULT_RECOMMENDATION_LOG_TOPK = 10


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
    item_labels: Optional[Dict[int, str]] = None,
    topk: int = DEFAULT_RECOMMENDATION_LOG_TOPK,
    num_users: int = DEFAULT_RECOMMENDATION_LOG_USERS,
) -> dict:
    """Build a compact, deterministic recommendation log from final rankings.

    Parameters
    ----------
    rankings : dict
        Output from ``Evaluator.rank`` containing ``user_ids``,
        ``topk_items``, and ``pos_index``.
    test_user_items : dict
        Mapping from user id to set of ground-truth item ids.
    item_labels : dict, optional
        Mapping from internal item id to a human-readable label
        (e.g. movie title, artist name). When *None*, items are
        logged by their numeric id only.
    topk : int
        Number of top recommendations to log per user.
    num_users : int
        Number of users to include in the log (first N by sorted id).
    """
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
            "records": [],
        }

    if item_labels is None:
        item_labels = {}

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
            top_recommendations.append(
                {
                    "item_id": int(item_id),
                    "label": item_labels.get(int(item_id)),
                    "rank": rank,
                    "is_hit": bool(is_hit),
                }
            )

        ground_truth = sorted(test_user_items.get(user_id, set()))
        records.append(
            {
                "user_id": int(user_id),
                "num_test_positives": len(ground_truth),
                "ground_truth_items": [
                    {"item_id": int(iid), "label": item_labels.get(int(iid))}
                    for iid in ground_truth
                ],
                "top_recommendations": top_recommendations,
                "num_hits_at_logged_k": int(sum(hit_flags)),
            }
        )

    return {
        "selection": "sorted_user_ids_first_n",
        "requested_user_count": selected_num_users,
        "logged_user_count": len(records),
        "topk": selected_topk,
        "records": records,
    }


def _item_display(item: dict) -> str:
    """Return a human-readable string for a single item."""
    label = item.get("label")
    if label:
        return label
    return f"item:{item['item_id']}"


def build_recommendation_summary_rows(recommendation_log: dict) -> list[dict]:
    """Return one compact row per user for CSV export."""
    rows = []
    for record in recommendation_log.get("records", []):
        recommendations = record.get("top_recommendations", [])
        hit_items = [rec for rec in recommendations if rec.get("is_hit")]
        rows.append(
            {
                "user_id": record["user_id"],
                "num_test_positives": record["num_test_positives"],
                "num_hits_at_logged_k": record["num_hits_at_logged_k"],
                "ground_truth": " | ".join(
                    _item_display(item)
                    for item in record.get("ground_truth_items", [])
                ),
                "hits": " | ".join(_item_display(item) for item in hit_items),
                "top_recommendations": " | ".join(
                    f"{rec['rank']}. {_item_display(rec)} "
                    f"({'hit' if rec.get('is_hit') else 'miss'})"
                    for rec in recommendations
                ),
            }
        )
    return rows


def build_recommendation_detail_rows(recommendation_log: dict) -> list[dict]:
    """Return one row per recommendation for CSV export."""
    rows = []
    for record in recommendation_log.get("records", []):
        ground_truth_str = " | ".join(
            _item_display(item) for item in record.get("ground_truth_items", [])
        )
        for rec in record.get("top_recommendations", []):
            rows.append(
                {
                    "user_id": record["user_id"],
                    "num_test_positives": record["num_test_positives"],
                    "num_hits_at_logged_k": record["num_hits_at_logged_k"],
                    "ground_truth": ground_truth_str,
                    "rank": rec["rank"],
                    "is_hit": rec["is_hit"],
                    "item_id": rec["item_id"],
                    "label": rec.get("label"),
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
