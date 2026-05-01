import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from run_experiments import _metrics_to_show, _quality_metrics_to_show, load_config
from utils.experiment_config import resolve_config


class ExperimentConfigTests(unittest.TestCase):
    def test_ml100k_compact_config_resolves_to_final_run_defaults(self):
        config = resolve_config({"dataset": "ml-100k"})

        self.assertEqual(config["dataset"], "ml-100k")
        self.assertEqual(config["epochs"], 50)
        self.assertEqual(config["learning_rate"], 0.0003)
        self.assertEqual(config["patience"], 8)
        self.assertEqual(config["train_batch_size"], 512)
        self.assertEqual(config["metrics"], ["Recall", "NDCG", "MRR", "Hit"])
        self.assertEqual(config["topk"], [5, 10, 20])
        self.assertEqual(config["valid_metric"], "NDCG@10")
        self.assertEqual(config["min_rating"], 4)
        self.assertFalse(config["implicit_feedback"])
        self.assertNotIn("benchmark_filename", config)

    def test_gowalla_compact_config_resolves_dataset_preset(self):
        config = resolve_config({"dataset": "gowalla-1m"})

        self.assertEqual(config["dataset"], "gowalla-1m")
        self.assertTrue(config["implicit_feedback"])
        self.assertIsNone(config["min_rating"])
        self.assertEqual(config["benchmark_filename"], ["train", "valid", "test"])
        self.assertEqual(config["epochs"], 30)
        self.assertEqual(config["patience"], 5)
        self.assertEqual(config["topk"], [20, 50])
        self.assertEqual(config["valid_metric"], "NDCG@20")
        self.assertEqual(config["train_batch_size"], 1024)
        self.assertEqual(config["candidate_pool_size"], 300)

    def test_config_overrides_win_over_defaults_and_presets(self):
        config = resolve_config(
            {
                "dataset": "ml-100k",
                "candidate_pool_size": 50,
                "feature_aware": True,
                "mixed_index_batch_size": 1024,
            }
        )

        self.assertEqual(config["candidate_pool_size"], 50)
        self.assertTrue(config["feature_aware"])
        self.assertEqual(config["mixed_index_batch_size"], 1024)

    def test_unknown_dataset_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported dataset 'ml-1m'"):
            resolve_config({"dataset": "ml-1m"})

    def test_missing_dataset_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Config must define a dataset"):
            resolve_config({})

    def test_unknown_config_key_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unknown config key"):
            resolve_config({"dataset": "ml-100k", "bogus": 1})

    def test_load_config_returns_resolved_config(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("dataset: gowalla-1m\n", encoding="utf-8")

            config = load_config(path)

        self.assertEqual(config["dataset"], "gowalla-1m")
        self.assertEqual(config["benchmark_filename"], ["train", "valid", "test"])
        self.assertEqual(config["valid_metric"], "NDCG@20")

    def test_all_experiment_configs_resolve(self):
        for path in sorted(Path("config/exp").rglob("*.yaml")):
            with self.subTest(config=str(path)):
                config = load_config(path)
                self.assertIn(config["dataset"], {"ml-100k", "gowalla-1m"})

    def test_gowalla_reporting_uses_configured_validation_k(self):
        config = resolve_config({"dataset": "gowalla-1m"})

        self.assertIn("ndcg@20", _metrics_to_show(config))
        self.assertIn("recall@20", _metrics_to_show(config))
        self.assertIn("item_coverage@20", _quality_metrics_to_show(config))


if __name__ == "__main__":
    unittest.main()
