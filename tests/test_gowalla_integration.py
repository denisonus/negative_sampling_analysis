import unittest

from scripts.prepare_gowalla_lightgcn import (
    merge_user_interactions,
    split_interactions,
    validate_split,
)
from utils.experiment_config import resolve_config


class GowallaConverterTests(unittest.TestCase):
    def test_internal_split_is_deterministic_and_disjoint(self):
        train = {
            0: [1, 2, 2, 3],
            1: [5],
            2: [8, 9],
        }
        test = {
            0: [3, 4, 5],
            1: [6],
            2: [10],
        }

        merged = merge_user_interactions([train, test])
        self.assertEqual(merged[0], [1, 2, 3, 4, 5])
        self.assertEqual(merged[1], [5, 6])
        self.assertEqual(merged[2], [8, 9, 10])

        first = split_interactions(merged, split=(0.8, 0.1, 0.1), seed=42)
        second = split_interactions(merged, split=(0.8, 0.1, 0.1), seed=42)

        self.assertEqual(first, second)
        validate_split(first)

        by_user = {split_name: {} for split_name in first}
        for split_name, rows in first.items():
            for user_id, item_id in rows:
                by_user[split_name].setdefault(user_id, set()).add(item_id)

        self.assertEqual(by_user["train"][1], {5, 6})
        self.assertNotIn(1, by_user["valid"])
        self.assertNotIn(1, by_user["test"])

        for user_id, item_ids in merged.items():
            if len(item_ids) >= 3:
                self.assertGreaterEqual(len(by_user["train"].get(user_id, set())), 1)
                self.assertGreaterEqual(len(by_user["test"].get(user_id, set())), 1)

class GowallaLoaderConfigTests(unittest.TestCase):
    def test_gowalla_config_uses_raw_lightgcn_inputs(self):
        resolved = resolve_config({"dataset": "gowalla-1m"})

        self.assertTrue(resolved["implicit_feedback"])
        self.assertIsNone(resolved["min_rating"])
        self.assertEqual(resolved["topk"], [20, 50])
        self.assertEqual(resolved["valid_metric"], "NDCG@20")


if __name__ == "__main__":
    unittest.main()
