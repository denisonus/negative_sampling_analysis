import tempfile
import unittest
from pathlib import Path

from scripts.prepare_gowalla_lightgcn import (
    INTER_HEADER,
    merge_user_interactions,
    split_interactions,
    validate_split,
    write_recbole_inter,
)
from utils.data_utils import build_recbole_config_dict


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

    def test_recbole_inter_writer_uses_expected_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gowalla-1m.train.inter"
            count = write_recbole_inter(path, [(3, 10), (4, 11)])

            self.assertEqual(count, 2)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [INTER_HEADER.strip(), "3\t10", "4\t11"],
            )


class GowallaLoaderConfigTests(unittest.TestCase):
    def test_explicit_movielens_config_keeps_rating_filter(self):
        config = build_recbole_config_dict(
            "ml-100k",
            data_path="dataset/",
            min_rating=4,
            feature_aware=False,
        )

        self.assertEqual(
            config["load_col"]["inter"],
            ["user_id", "item_id", "rating", "timestamp"],
        )
        self.assertEqual(config["val_interval"], {"rating": "[4,inf)"})
        self.assertNotIn("benchmark_filename", config)
        self.assertEqual(config["eval_args"]["order"], "TO")

    def test_implicit_gowalla_config_uses_benchmark_files(self):
        config = build_recbole_config_dict(
            "gowalla-1m",
            data_path="dataset/",
            min_rating=None,
            implicit_feedback=True,
            benchmark_filename=["train", "valid", "test"],
        )

        self.assertEqual(config["load_col"]["inter"], ["user_id", "item_id"])
        self.assertNotIn("val_interval", config)
        self.assertEqual(config["benchmark_filename"], ["train", "valid", "test"])
        self.assertEqual(config["eval_args"]["order"], "RO")


if __name__ == "__main__":
    unittest.main()
