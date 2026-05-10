import unittest
from typing import cast

import torch

from models import TwoTowerModel
from samplers import get_sampler
from samplers.hard import HardNegativeSampler


class TwoTowerModelTests(unittest.TestCase):
    def test_id_only_mode_preserves_shapes_and_loss_behavior(self):
        model = TwoTowerModel(
            num_users=5,
            num_items=7,
            embedding_size=8,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
        )
        user_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        pos_item_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        neg_item_ids = torch.tensor([[4, 5], [5, 6], [0, 6]], dtype=torch.long)

        loss = model.compute_loss(user_ids, pos_item_ids, neg_item_ids)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.get_user_embedding(user_ids).shape, (3, 8))
        self.assertEqual(model.get_item_embedding(pos_item_ids).shape, (3, 8))
        self.assertEqual(model.get_item_embedding(neg_item_ids).shape, (3, 2, 8))
        self.assertEqual(model.predict(user_ids).shape, (3, 7))

    def test_hard_sampler_works_with_id_only_model(self):
        model = TwoTowerModel(
            num_users=4,
            num_items=8,
            embedding_size=8,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
        )
        user_item_dict = {
            0: {1, 2},
            1: {2, 3},
            2: {4},
            3: {5},
        }
        sampler = cast(
            HardNegativeSampler,
            get_sampler(
                "hard",
                num_items=8,
                num_neg_samples=2,
                user_item_dict=user_item_dict,
                model=model,
                candidate_pool_size=4,
            ),
        )

        neg_items = sampler.sample(
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([1, 2, 4], dtype=torch.long),
        )

        self.assertEqual(neg_items.shape, (3, 2))
        for row_idx, user_id in enumerate([0, 1, 2]):
            positives = user_item_dict[user_id]
            for item_id in neg_items[row_idx].tolist():
                self.assertNotIn(item_id, positives)


if __name__ == "__main__":
    unittest.main()
