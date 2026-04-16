import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analysis import (
    plot_metric_by_k,
    plot_multi_metric_sweep,
    plot_user_bucket_delta_heatmap,
    plot_user_bucket_metrics,
    save_competitive_summary,
    save_summary_table,
    save_user_bucket_metrics_table,
)
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
            tail_ratio=0.4,
            pair_sample_cap=10,
        )

        probabilities = popularity / popularity.sum()
        expected_novelty = float((-np.log2(probabilities[topk_items])).mean())
        expected_personalization = 1.0 - np.mean([1 / 3, 0.0, 0.0])

        self.assertAlmostEqual(metrics["item_coverage@2"], 1.0)
        self.assertAlmostEqual(metrics["novelty@2"], expected_novelty)
        self.assertAlmostEqual(metrics["tail_percentage@2"], 2 / 6)
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
                        "tail_percentage@10": 0.2,
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
                        "tail_percentage@10": 0.4,
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

    def test_user_bucket_analysis_outputs_are_created(self):
        results = {
            "statistics": {
                "hard": {
                    "metrics": {"ndcg@10": {"mean": 0.2, "std": 0.01}},
                    "bucket_metrics": {
                        "1-5": {
                            "ndcg@10": {
                                "mean": 0.15,
                                "std": 0.01,
                                "ci_lower": 0.14,
                                "ci_upper": 0.16,
                                "values": [0.15, 0.15],
                            },
                            "recall@10": {
                                "mean": 0.25,
                                "std": 0.01,
                                "ci_lower": 0.24,
                                "ci_upper": 0.26,
                                "values": [0.25, 0.25],
                            },
                            "hit@10": {
                                "mean": 0.35,
                                "std": 0.01,
                                "ci_lower": 0.34,
                                "ci_upper": 0.36,
                                "values": [0.35, 0.35],
                            },
                            "mrr@10": {
                                "mean": 0.45,
                                "std": 0.01,
                                "ci_lower": 0.44,
                                "ci_upper": 0.46,
                                "values": [0.45, 0.45],
                            },
                        },
                        "6-20": {
                            "ndcg@10": {
                                "mean": 0.18,
                                "std": 0.02,
                                "ci_lower": 0.16,
                                "ci_upper": 0.20,
                                "values": [0.18, 0.18],
                            },
                            "recall@10": {
                                "mean": 0.28,
                                "std": 0.02,
                                "ci_lower": 0.26,
                                "ci_upper": 0.30,
                                "values": [0.28, 0.28],
                            },
                            "hit@10": {
                                "mean": 0.38,
                                "std": 0.02,
                                "ci_lower": 0.36,
                                "ci_upper": 0.40,
                                "values": [0.38, 0.38],
                            },
                            "mrr@10": {
                                "mean": 0.48,
                                "std": 0.02,
                                "ci_lower": 0.46,
                                "ci_upper": 0.50,
                                "values": [0.48, 0.48],
                            },
                        },
                    },
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "user_bucket_metrics.csv"
            plot_path = Path(tmpdir) / "user_bucket_metrics.png"
            heatmap_path = Path(tmpdir) / "user_bucket_ndcg10_delta_heatmap.png"

            rows = save_user_bucket_metrics_table(results, csv_path)
            plotted_buckets = plot_user_bucket_metrics(results, output_path=plot_path)
            heatmap = plot_user_bucket_delta_heatmap(
                results,
                baseline="uniform",
                output_path=heatmap_path,
            )

            self.assertTrue(rows)
            self.assertEqual(plotted_buckets, ["1-5", "6-20"])
            self.assertTrue(any(row["metric"] == "mrr@10" for row in rows))
            self.assertIsNone(heatmap)
            self.assertTrue(csv_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertFalse(heatmap_path.exists())

    def test_summary_and_metric_by_k_outputs_include_recall20(self):
        results = {
            "statistics": {
                "uniform": {
                    "metrics": {
                        "ndcg@5": {"mean": 0.10},
                        "ndcg@10": {"mean": 0.12},
                        "ndcg@20": {"mean": 0.14},
                        "recall@5": {"mean": 0.20},
                        "recall@10": {"mean": 0.25},
                        "recall@20": {"mean": 0.30},
                        "mrr@5": {"mean": 0.30},
                        "mrr@10": {"mean": 0.32},
                        "mrr@20": {"mean": 0.33},
                        "hit@10": {"mean": 0.40},
                        "precision@10": {"mean": 0.08},
                        "map@10": {"mean": 0.09},
                    },
                    "quality_metrics": {
                        "item_coverage@10": {"mean": 0.5},
                        "novelty@10": {"mean": 1.2},
                        "tail_percentage@10": {"mean": 0.1},
                        "personalization@10": {"mean": 0.8},
                    },
                    "timing": {
                        "total_time": {"mean": 1.0},
                        "sampling_time": {"mean": 0.2},
                        "training_time": {"mean": 0.8},
                    },
                },
                "hard": {
                    "metrics": {
                        "ndcg@5": {"mean": 0.11},
                        "ndcg@10": {"mean": 0.13},
                        "ndcg@20": {"mean": 0.15},
                        "recall@5": {"mean": 0.21},
                        "recall@10": {"mean": 0.27},
                        "recall@20": {"mean": 0.31},
                        "mrr@5": {"mean": 0.29},
                        "mrr@10": {"mean": 0.31},
                        "mrr@20": {"mean": 0.32},
                        "hit@10": {"mean": 0.41},
                        "precision@10": {"mean": 0.07},
                        "map@10": {"mean": 0.10},
                    },
                    "quality_metrics": {
                        "item_coverage@10": {"mean": 0.6},
                        "novelty@10": {"mean": 1.1},
                        "tail_percentage@10": {"mean": 0.2},
                        "personalization@10": {"mean": 0.7},
                    },
                    "timing": {
                        "total_time": {"mean": 2.0},
                        "sampling_time": {"mean": 0.5},
                        "training_time": {"mean": 1.5},
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary_metrics.csv"
            competitive_path = Path(tmpdir) / "competitive_summary.csv"
            metric_by_k_path = Path(tmpdir) / "metric_by_k.png"

            save_summary_table(results, summary_path, metadata={"config": {"feature_aware": False}})
            rows = save_competitive_summary(results, competitive_path)
            plotted = plot_metric_by_k(results, output_path=metric_by_k_path)

            header = summary_path.read_text().splitlines()[0].split(",")
            self.assertIn("recall@20", header)
            self.assertIn("precision@10", header)
            self.assertIn("map@10", header)
            self.assertTrue(rows)
            self.assertIn("recall@20", rows[0])
            self.assertEqual(plotted, ["ndcg", "recall", "mrr"])
            self.assertTrue(metric_by_k_path.exists())

    def test_multi_metric_sweep_outputs_are_created(self):
        bundle_results = {
            "statistics": {
                "uniform": {
                    "metrics": {
                        "ndcg@10": {"mean": 0.10},
                        "recall@10": {"mean": 0.20},
                        "recall@20": {"mean": 0.30},
                        "mrr@10": {"mean": 0.40},
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            results_files = []
            for idx, num_neg in enumerate([1, 4], start=1):
                run_dir = Path(tmpdir) / f"run_{idx}"
                run_dir.mkdir()
                (run_dir / "results.json").write_text(json.dumps(bundle_results))
                metadata = {
                    "config": {
                        "num_neg_samples": num_neg,
                        "topk": [5, 10, 20],
                        "valid_metric": "NDCG@10",
                    }
                }
                (run_dir / "metadata.json").write_text(json.dumps(metadata))
                results_files.append(str(run_dir / "results.json"))

            plot_path = Path(tmpdir) / "parameter_sweep_core_metrics.png"
            csv_path = Path(tmpdir) / "parameter_sweep_core_metrics.csv"
            rows = plot_multi_metric_sweep(
                results_files,
                strategies=["uniform"],
                output_path=plot_path,
                csv_path=csv_path,
            )

            self.assertTrue(rows)
            self.assertTrue(plot_path.exists())
            self.assertTrue(csv_path.exists())
            csv_text = csv_path.read_text()
            self.assertIn("metric_name", csv_text)
            self.assertIn("recall@20", csv_text)


if __name__ == "__main__":
    unittest.main()
