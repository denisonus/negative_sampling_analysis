import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis import generate_full_report
from evaluation.evaluator import Evaluator
from evaluation.quality import compute_quality_metrics
from run_experiments import compute_statistics, save_results


class StaticScoreModel(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32))

    def get_all_item_embeddings(self):
        return torch.empty(self.scores.size(1), 1, device=self.scores.device)

    def predict(self, user_ids, item_ids=None, all_item_emb=None):
        return self.scores[user_ids]


class DummyTestData:
    def __init__(self, uid_list, uid2positive_item, uid2history_item):
        self.uid_list = torch.tensor(uid_list, dtype=torch.long)
        self.uid2positive_item = uid2positive_item
        self.uid2history_item = uid2history_item


class QualityMetricsTests(unittest.TestCase):
    def test_quality_metrics_match_expected_values(self):
        topk_items = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int64)
        popularity = np.array([10, 5, 2, 1, 1], dtype=np.float64)

        metrics = compute_quality_metrics(
            topk_items,
            item_popularity=popularity,
            num_items=5,
            topk=[2],
            seed=7,
            pair_sample_cap=10,
        )

        probabilities = popularity / popularity.sum()
        expected_novelty = float((-np.log2(probabilities[topk_items])).mean())
        expected_avg_popularity = float(popularity[topk_items].mean())
        expected_personalization = 1.0 - np.mean([1 / 3, 0.0, 0.0])

        self.assertAlmostEqual(metrics["item_coverage@2"], 1.0)
        self.assertAlmostEqual(metrics["novelty@2"], expected_novelty)
        self.assertAlmostEqual(metrics["avg_popularity@2"], expected_avg_popularity)
        self.assertAlmostEqual(
            metrics["personalization@2"], expected_personalization
        )

    def test_rank_masks_history_items(self):
        evaluator = Evaluator(num_items=4, metrics=["Recall", "NDCG"], topk=[2])
        model = StaticScoreModel(
            [
                [0.95, 0.90, 0.20, 0.10],
                [0.20, 0.80, 0.70, 0.60],
            ]
        )
        test_data = DummyTestData(
            uid_list=[0, 1],
            uid2positive_item={0: np.array([1]), 1: np.array([3])},
            uid2history_item={0: np.array([0]), 1: np.array([1])},
        )

        rankings = evaluator.rank(model, test_data)

        self.assertNotIn(0, rankings["topk_items"][0].tolist())
        self.assertNotIn(1, rankings["topk_items"][1].tolist())
        self.assertEqual(rankings["topk_items"].shape, (2, 2))

    def test_user_bucket_metrics_match_expected_values(self):
        evaluator = Evaluator(
            num_items=20, metrics=["Recall", "NDCG", "Hit", "MRR"], topk=[10]
        )
        rankings = {
            "user_ids": np.array([10, 11, 12, 13], dtype=np.int64),
            "topk_items": np.zeros((4, 10), dtype=np.int64),
            "pos_index": np.array(
                [
                    [False, False, True, False, False, False, False, False, False, False],
                    [True, False, False, False, False, False, False, False, False, False],
                    [False, True, False, False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False, False, False],
                ],
                dtype=bool,
            ),
            "pos_len": np.array([1, 1, 1, 1], dtype=np.int64),
        }

        bucket_metrics = evaluator.evaluate_user_buckets_from_rankings(
            rankings,
            user_train_counts={10: 0, 11: 2, 12: 7, 13: 25},
        )

        self.assertEqual(list(bucket_metrics.keys()), ["0", "1-5", "6-20", "21+"])
        self.assertAlmostEqual(bucket_metrics["0"]["ndcg@10"], 1.0 / np.log2(4))
        self.assertAlmostEqual(bucket_metrics["0"]["recall@10"], 1.0)
        self.assertAlmostEqual(bucket_metrics["0"]["hit@10"], 1.0)
        self.assertAlmostEqual(bucket_metrics["0"]["mrr@10"], 1.0 / 3.0)
        self.assertAlmostEqual(bucket_metrics["1-5"]["ndcg@10"], 1.0)
        self.assertAlmostEqual(bucket_metrics["1-5"]["mrr@10"], 1.0)
        self.assertAlmostEqual(bucket_metrics["6-20"]["ndcg@10"], 1.0 / np.log2(3))
        self.assertAlmostEqual(bucket_metrics["6-20"]["mrr@10"], 0.5)
        self.assertAlmostEqual(bucket_metrics["21+"]["ndcg@10"], 0.0)
        self.assertAlmostEqual(bucket_metrics["21+"]["recall@10"], 0.0)
        self.assertAlmostEqual(bucket_metrics["21+"]["hit@10"], 0.0)
        self.assertAlmostEqual(bucket_metrics["21+"]["mrr@10"], 0.0)

    def test_quality_metrics_are_aggregated_and_saved(self):
        all_results = {
            "uniform": [
                {
                    "seed": 42,
                    "test_metrics": {"ndcg@10": 0.1, "recall@10": 0.2},
                    "bucket_metrics": {
                        "1-5": {"ndcg@10": 0.08, "recall@10": 0.12, "hit@10": 0.2},
                        "6-20": {"ndcg@10": 0.14, "recall@10": 0.25, "hit@10": 0.35},
                    },
                    "quality_metrics": {
                        "item_coverage@10": 0.3,
                        "novelty@10": 1.1,
                        "avg_popularity@10": 5.2,
                        "personalization@10": 0.7,
                    },
                    "timing": {
                        "total_time": 1.0,
                        "total_sampling_time": 0.3,
                        "total_training_time": 0.7,
                    },
                    "train_history": {
                        "train_losses": [1.0],
                        "valid_metrics": [{"ndcg@10": 0.1}],
                        "best_epoch": 0,
                        "epoch_times": [1.0],
                        "sampling_times": [0.3],
                        "training_times": [0.7],
                    },
                    "dataset_stats": {"num_users": 2, "num_items": 5, "num_train_interactions": 4},
                },
                {
                    "seed": 1042,
                    "test_metrics": {"ndcg@10": 0.2, "recall@10": 0.3},
                    "bucket_metrics": {
                        "1-5": {"ndcg@10": 0.12, "recall@10": 0.18, "hit@10": 0.24},
                        "6-20": {"ndcg@10": 0.22, "recall@10": 0.32, "hit@10": 0.42},
                    },
                    "quality_metrics": {
                        "item_coverage@10": 0.5,
                        "novelty@10": 1.5,
                        "avg_popularity@10": 4.8,
                        "personalization@10": 0.9,
                    },
                    "timing": {
                        "total_time": 2.0,
                        "total_sampling_time": 0.5,
                        "total_training_time": 1.5,
                    },
                    "train_history": {
                        "train_losses": [0.9],
                        "valid_metrics": [{"ndcg@10": 0.2}],
                        "best_epoch": 0,
                        "epoch_times": [2.0],
                        "sampling_times": [0.5],
                        "training_times": [1.5],
                    },
                    "dataset_stats": {"num_users": 2, "num_items": 5, "num_train_interactions": 4},
                },
            ]
        }

        stats = compute_statistics(all_results)
        self.assertIn("quality_metrics", stats["uniform"])
        self.assertIn("bucket_metrics", stats["uniform"])
        self.assertAlmostEqual(
            stats["uniform"]["quality_metrics"]["item_coverage@10"]["mean"], 0.4
        )
        self.assertAlmostEqual(
            stats["uniform"]["metrics"]["ndcg@10"]["mean"], 0.15
        )
        self.assertAlmostEqual(
            stats["uniform"]["bucket_metrics"]["1-5"]["ndcg@10"]["mean"], 0.10
        )
        self.assertAlmostEqual(
            stats["uniform"]["bucket_metrics"]["6-20"]["hit@10"]["mean"], 0.385
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _, run_dir = save_results(
                all_results,
                output_dir=tmpdir,
                config={"epochs": 5, "device": "cpu"},
            )
            results_path = Path(run_dir) / "results.json"
            saved = json.loads(results_path.read_text())

        raw_run = saved["raw_results"]["uniform"][0]
        self.assertIn("bucket_metrics", raw_run)
        self.assertIn("1-5", raw_run["bucket_metrics"])
        self.assertIn("quality_metrics", raw_run)
        self.assertIn("item_coverage@10", raw_run["quality_metrics"])
        self.assertIn("bucket_metrics", saved["statistics"]["uniform"])
        self.assertIn("1-5", saved["statistics"]["uniform"]["bucket_metrics"])
        self.assertIn("quality_metrics", saved["statistics"]["uniform"])
        self.assertIn(
            "novelty@10", saved["statistics"]["uniform"]["quality_metrics"]
        )

    def test_generate_full_report_uses_lean_outputs(self):
        def strategy_stats(ndcg_values, offset):
            return {
                "metrics": {
                    "ndcg@5": {"mean": ndcg_values[0] - 0.01},
                    "ndcg@10": {
                        "mean": sum(ndcg_values) / len(ndcg_values),
                        "std": 0.01,
                        "values": ndcg_values,
                    },
                    "ndcg@20": {"mean": ndcg_values[-1] + 0.01},
                    "recall@5": {"mean": 0.20 + offset},
                    "recall@10": {"mean": 0.25 + offset},
                    "recall@20": {"mean": 0.30 + offset},
                    "mrr@5": {"mean": 0.30 + offset},
                    "mrr@10": {"mean": 0.32 + offset},
                    "mrr@20": {"mean": 0.33 + offset},
                    "hit@10": {"mean": 0.40 + offset},
                },
                "quality_metrics": {
                    "item_coverage@10": {"mean": 0.5 + offset, "std": 0.01},
                    "novelty@10": {"mean": 1.2 + offset, "std": 0.01},
                    "avg_popularity@10": {"mean": 5.0 - offset, "std": 0.1},
                    "personalization@10": {"mean": 0.8 - offset, "std": 0.01},
                },
                "timing": {
                    "total_time": {"mean": 2.0 + offset},
                    "sampling_time": {"mean": 0.5 + offset},
                    "training_time": {"mean": 1.5},
                },
                "bucket_metrics": {
                    "1-5": {
                        "ndcg@10": {
                            "mean": 0.10 + offset,
                            "std": 0.01,
                            "values": [0.10 + offset, 0.11 + offset],
                        },
                        "recall@10": {"mean": 0.20 + offset, "values": [0.20 + offset]},
                        "hit@10": {"mean": 0.30 + offset, "values": [0.30 + offset]},
                        "mrr@10": {"mean": 0.40 + offset, "values": [0.40 + offset]},
                    },
                    "6-20": {
                        "ndcg@10": {
                            "mean": 0.15 + offset,
                            "std": 0.01,
                            "values": [0.15 + offset, 0.16 + offset],
                        },
                        "recall@10": {"mean": 0.25 + offset, "values": [0.25 + offset]},
                        "hit@10": {"mean": 0.35 + offset, "values": [0.35 + offset]},
                        "mrr@10": {"mean": 0.45 + offset, "values": [0.45 + offset]},
                    },
                },
            }

        results = {
            "statistics": {
                "uniform": strategy_stats([0.10, 0.12], 0.0),
                "hard": strategy_stats([0.13, 0.15], 0.05),
            },
            "raw_results": {
                "uniform": [
                    {"valid_metrics": [{"ndcg@10": 0.08}, {"ndcg@10": 0.12}], "train_losses": [1.5, 1.2]},
                    {"valid_metrics": [{"ndcg@10": 0.09}, {"ndcg@10": 0.11}], "train_losses": [1.4, 1.1]},
                ],
                "hard": [
                    {"valid_metrics": [{"ndcg@10": 0.10}, {"ndcg@10": 0.15}], "train_losses": [1.3, 0.9]},
                    {"valid_metrics": [{"ndcg@10": 0.11}, {"ndcg@10": 0.14}], "train_losses": [1.2, 0.8]},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            output_dir = Path(tmpdir) / "analysis"
            run_dir.mkdir()
            results_path = run_dir / "results.json"
            metadata_path = run_dir / "metadata.json"
            results_path.write_text(json.dumps(results))
            metadata_path.write_text(json.dumps({"config": {"feature_aware": False}}))

            generate_full_report(str(results_path), output_dir=str(output_dir))

            expected_files = [
                "summary_metrics.csv",
                "relative_improvements_vs_uniform.csv",
                "dashboard.png",
                "metric_by_k.png",
                "quality_metrics.png",
                "user_bucket_metrics.csv",
                "user_bucket_ndcg10_delta_heatmap.png",
                "ndcg10_delta_vs_uniform.png",
                "training_dynamics.png",
                "significance_ndcg10.csv",
            ]
            omitted_files = [
                "relevance_metrics.png",
                "ndcg_vs_time.png",
                "ndcg_vs_coverage.png",
                "ndcg_vs_novelty.png",
                "competitive_quality.png",
                "competitive_summary.csv",
                "user_bucket_metrics.png",
                "feature_quality_tradeoff.png",
            ]

            for filename in expected_files:
                self.assertTrue((output_dir / filename).exists(), filename)
            for filename in omitted_files:
                self.assertFalse((output_dir / filename).exists(), filename)

            with (output_dir / "relative_improvements_vs_uniform.csv").open(
                newline=""
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))

            ndcg_row = next(
                row
                for row in rows
                if row["strategy"] == "hard" and row["metric"] == "ndcg@10"
            )
            self.assertEqual(ndcg_row["baseline"], "uniform")
            self.assertEqual(ndcg_row["metric_type"], "relevance")
            self.assertAlmostEqual(float(ndcg_row["baseline_value"]), 0.11)
            self.assertAlmostEqual(float(ndcg_row["strategy_value"]), 0.14)
            self.assertAlmostEqual(float(ndcg_row["absolute_delta"]), 0.03)
            self.assertAlmostEqual(
                float(ndcg_row["relative_improvement_percent"]),
                (0.03 / 0.11) * 100,
            )


if __name__ == "__main__":
    unittest.main()
