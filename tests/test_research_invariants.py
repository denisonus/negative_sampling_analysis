import unittest
from typing import cast

import numpy as np
import torch

from models import TwoTowerModel
from run_experiments import set_seed
from samplers import get_sampler
from samplers.base import SamplingResult
from samplers.curriculum import CurriculumNegativeSampler
from samplers.debiased import DebiasedNegativeSampler
from samplers.dns import DNSNegativeSampler
from samplers.hard import HardNegativeSampler
from samplers.mixed import MixedHardUniformNegativeSampler
from samplers.mixed_in_batch_uniform import MixedInBatchUniformNegativeSampler
from samplers.popularity import PopularityNegativeSampler
from utils.experiment_config import resolve_config
from utils.trainer import Trainer, _select_validation_log_metrics


class DummyEvaluator:
    def __init__(self, metrics):
        self.metrics = list(metrics)
        self.index = 0

    def evaluate(self, model, valid_loader):
        metric = self.metrics[self.index]
        self.index += 1
        return {"ndcg@10": metric}


class ScriptedTrainer(Trainer):
    def __init__(self, model, sampler, config, device, scripted_weights):
        super().__init__(model, sampler, config, device)
        self.scripted_weights = scripted_weights

    def train_epoch(self, train_loader, epoch=0):
        with torch.no_grad():
            self.model.weight.fill_(self.scripted_weights[epoch])

        self.train_losses.append(float(epoch))
        self.sampling_times.append(0.0)
        self.training_times.append(0.0)
        return float(epoch)


class ResearchInvariantTests(unittest.TestCase):
    def setUp(self):
        self.num_users = 4
        self.num_items = 20
        self.num_neg_samples = 4
        self.user_item_dict = {
            0: {1, 2, 3},
            1: {3, 4, 5},
            2: {0, 6},
            3: {7, 8},
        }
        self.user_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        self.pos_item_ids = torch.tensor([1, 3, 6, 7], dtype=torch.long)
        self.item_popularity = np.arange(1, self.num_items + 1, dtype=np.float64)

    def _build_model(self):
        return TwoTowerModel(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_size=8,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
        )

    def _extract_neg_items(
        self, sample_result: torch.Tensor | SamplingResult
    ) -> torch.Tensor:
        if isinstance(sample_result, SamplingResult):
            return sample_result.neg_items
        return sample_result

    def test_seed_reset_reproduces_model_parameters_and_samples(self):
        set_seed(123)
        model_a = self._build_model()
        sampler_a = get_sampler(
            "uniform",
            num_items=self.num_items,
            num_neg_samples=self.num_neg_samples,
            user_item_dict=self.user_item_dict,
        )
        samples_a = self._extract_neg_items(
            sampler_a.sample(self.user_ids, self.pos_item_ids)
        )
        state_a = {
            name: tensor.detach().clone()
            for name, tensor in model_a.state_dict().items()
        }

        set_seed(123)
        model_b = self._build_model()
        sampler_b = get_sampler(
            "uniform",
            num_items=self.num_items,
            num_neg_samples=self.num_neg_samples,
            user_item_dict=self.user_item_dict,
        )
        samples_b = self._extract_neg_items(
            sampler_b.sample(self.user_ids, self.pos_item_ids)
        )
        state_b = {
            name: tensor.detach().clone()
            for name, tensor in model_b.state_dict().items()
        }

        self.assertEqual(state_a.keys(), state_b.keys())
        for name in state_a:
            self.assertTrue(torch.equal(state_a[name], state_b[name]), msg=name)
        self.assertTrue(torch.equal(samples_a, samples_b))

    def test_explicit_samplers_never_return_known_positives(self):
        strategies = [
            "uniform",
            "popularity",
            "hard",
            "mixed_hard_uniform",
            "dns",
            "curriculum",
            "debiased",
        ]
        model = self._build_model()

        for strategy in strategies:
            with self.subTest(strategy=strategy):
                sampler = get_sampler(
                    strategy,
                    num_items=self.num_items,
                    num_neg_samples=self.num_neg_samples,
                    user_item_dict=self.user_item_dict,
                    item_popularity=self.item_popularity,
                    model=model,
                    candidate_pool_size=8,
                    hard_ratio=0.5,
                    dns_temperature=0.2,
                    curriculum_start_ratio=0.0,
                    curriculum_end_ratio=0.75,
                    curriculum_warmup_epochs=5,
                    tau_plus=0.15,
                )
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(2)
                if hasattr(sampler, "set_model"):
                    sampler.set_model(model)

                neg_items = self._extract_neg_items(
                    sampler.sample(self.user_ids, self.pos_item_ids)
                )
                self.assertEqual(
                    neg_items.shape,
                    (self.user_ids.size(0), self.num_neg_samples),
                )

                for row_idx, user_id in enumerate(self.user_ids.tolist()):
                    positives = self.user_item_dict[user_id]
                    for item_id in neg_items[row_idx].tolist():
                        self.assertNotIn(item_id, positives)

    def test_unique_negative_samplers_return_distinct_items_per_example(self):
        strategies = [
            "uniform",
            "hard",
            "mixed_hard_uniform",
            "dns",
            "curriculum",
            "debiased",
        ]
        model = self._build_model()

        for strategy in strategies:
            with self.subTest(strategy=strategy):
                sampler = get_sampler(
                    strategy,
                    num_items=self.num_items,
                    num_neg_samples=self.num_neg_samples,
                    user_item_dict=self.user_item_dict,
                    model=model,
                    candidate_pool_size=8,
                    hard_ratio=0.5,
                    dns_temperature=0.2,
                    curriculum_start_ratio=0.0,
                    curriculum_end_ratio=0.75,
                    curriculum_warmup_epochs=5,
                    tau_plus=0.15,
                )
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(2)
                if hasattr(sampler, "set_model"):
                    sampler.set_model(model)

                neg_items = self._extract_neg_items(
                    sampler.sample(self.user_ids, self.pos_item_ids)
                )
                for row in neg_items.tolist():
                    self.assertEqual(len(row), len(set(row)))

    def test_sampler_factory_applies_research_critical_kwargs(self):
        model = self._build_model()

        popularity = cast(
            PopularityNegativeSampler,
            get_sampler(
                "popularity",
                num_items=self.num_items,
                num_neg_samples=self.num_neg_samples,
                user_item_dict=self.user_item_dict,
                item_popularity=self.item_popularity,
                logq_correction=False,
                smoothing=0.5,
            ),
        )
        hard = cast(
            HardNegativeSampler,
            get_sampler(
                "hard",
                num_items=self.num_items,
                num_neg_samples=self.num_neg_samples,
                user_item_dict=self.user_item_dict,
                model=model,
                candidate_pool_size=7,
            ),
        )
        dns = cast(
            DNSNegativeSampler,
            get_sampler(
                "dns",
                num_items=self.num_items,
                num_neg_samples=self.num_neg_samples,
                user_item_dict=self.user_item_dict,
                model=model,
                candidate_pool_size=9,
                dns_temperature=0.3,
            ),
        )
        curriculum = cast(
            CurriculumNegativeSampler,
            get_sampler(
                "curriculum",
                num_items=self.num_items,
                num_neg_samples=self.num_neg_samples,
                user_item_dict=self.user_item_dict,
                model=model,
                candidate_pool_size=11,
                curriculum_start_ratio=0.1,
                curriculum_end_ratio=0.9,
                curriculum_warmup_epochs=7,
            ),
        )
        debiased = cast(
            DebiasedNegativeSampler,
            get_sampler(
                "debiased",
                num_items=self.num_items,
                num_neg_samples=self.num_neg_samples,
                user_item_dict=self.user_item_dict,
                tau_plus=0.2,
            ),
        )
        mixed_in_batch_uniform = cast(
            MixedInBatchUniformNegativeSampler,
            get_sampler(
                "mixed_in_batch_uniform",
                num_items=self.num_items,
                num_neg_samples=self.num_neg_samples,
                user_item_dict=self.user_item_dict,
                train_batch_size=32,
            ),
        )

        self.assertFalse(popularity.logq_correction)
        self.assertEqual(hard.candidate_pool_size, 7)
        self.assertEqual(dns.candidate_pool_size, 9)
        self.assertAlmostEqual(dns.temperature, 0.3)
        self.assertEqual(curriculum.candidate_pool_size, 11)
        self.assertAlmostEqual(curriculum.start_hard_ratio, 0.1)
        self.assertAlmostEqual(curriculum.end_hard_ratio, 0.9)
        self.assertEqual(curriculum.warmup_epochs, 7)
        self.assertAlmostEqual(debiased.tau_plus, 0.2)
        self.assertEqual(mixed_in_batch_uniform.name, "mixed_in_batch_uniform")
        self.assertEqual(mixed_in_batch_uniform.index_batch_size, 32)

    def test_fractional_hard_ratio_does_not_collapse_for_single_negative(self):
        model = self._build_model()

        mixed = cast(
            MixedHardUniformNegativeSampler,
            get_sampler(
                "mixed_hard_uniform",
                num_items=self.num_items,
                num_neg_samples=1,
                user_item_dict=self.user_item_dict,
                model=model,
                hard_ratio=0.5,
                candidate_pool_size=8,
            ),
        )
        curriculum = cast(
            CurriculumNegativeSampler,
            get_sampler(
                "curriculum",
                num_items=self.num_items,
                num_neg_samples=1,
                user_item_dict=self.user_item_dict,
                model=model,
                curriculum_start_ratio=0.5,
                curriculum_end_ratio=0.5,
                curriculum_warmup_epochs=1,
                candidate_pool_size=8,
            ),
        )

        set_seed(123)
        mixed_counts = [mixed._split_negative_budget(mixed.hard_ratio)[0] for _ in range(200)]
        set_seed(123)
        curriculum_counts = [
            curriculum._split_negative_budget(curriculum._get_current_hard_ratio())[0]
            for _ in range(200)
        ]

        for counts in [mixed_counts, curriculum_counts]:
            self.assertIn(0, counts)
            self.assertIn(1, counts)
            self.assertGreater(np.mean(counts), 0.35)
            self.assertLess(np.mean(counts), 0.65)

    def test_legacy_mixed_alias_maps_to_hard_uniform_variant(self):
        sampler = get_sampler(
            "mixed",
            num_items=self.num_items,
            num_neg_samples=self.num_neg_samples,
            user_item_dict=self.user_item_dict,
            model=self._build_model(),
        )

        self.assertEqual(sampler.name, "mixed_hard_uniform")

    def test_trainer_restores_best_checkpoint_after_early_stopping(self):
        model = torch.nn.Linear(1, 1, bias=False)
        sampler = get_sampler(
            "uniform",
            num_items=5,
            num_neg_samples=1,
            user_item_dict={0: {0}},
        )
        trainer = ScriptedTrainer(
            model=model,
            sampler=sampler,
            config=resolve_config(
                {"dataset": "ml-100k", "patience": 1, "valid_metric": "ndcg@10"}
            ),
            device="cpu",
            scripted_weights=[1.0, 2.0, 3.0],
        )
        evaluator = DummyEvaluator([0.8, 0.1, 0.0])

        history = trainer.fit(
            train_loader=[],
            valid_loader=[],
            evaluator=evaluator,
            epochs=3,
        )

        self.assertEqual(history["best_epoch"], 0)
        self.assertAlmostEqual(history["best_metric"], 0.8)
        self.assertAlmostEqual(model.weight.item(), 1.0)

    def test_validation_log_metrics_follow_configured_k(self):
        metrics = {
            "recall@20": 0.11,
            "ndcg@20": 0.22,
            "mrr@20": 0.33,
            "hit@20": 0.44,
            "recall@50": 0.55,
            "ndcg@50": 0.66,
        }

        logged = _select_validation_log_metrics(metrics, "NDCG@20")

        self.assertEqual(
            logged,
            {
                "recall@20": 0.11,
                "ndcg@20": 0.22,
                "mrr@20": 0.33,
                "hit@20": 0.44,
            },
        )


if __name__ == "__main__":
    unittest.main()
