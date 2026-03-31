import unittest
from functools import lru_cache

import torch

from models import Tower, TwoTowerModel
from samplers import get_sampler
from utils import extract_feature_data, load_recbole_dataset


@lru_cache(maxsize=2)
def _load_ml100k(feature_aware):
    return load_recbole_dataset(
        "ml-100k",
        data_path="dataset/",
        min_rating=4,
        feature_aware=feature_aware,
    )


def _require_ml100k(feature_aware):
    try:
        return _load_ml100k(feature_aware)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise unittest.SkipTest(f"ml-100k dataset unavailable: {exc}") from exc


class FeatureAwareLoaderTests(unittest.TestCase):
    def test_loader_only_loads_side_features_when_enabled(self):
        _, plain_dataset, _, _, _ = _require_ml100k(feature_aware=False)
        _, feature_dataset, _, _, _ = _require_ml100k(feature_aware=True)

        self.assertIsNone(plain_dataset.user_feat)
        self.assertIsNone(plain_dataset.item_feat)
        self.assertEqual(
            list(feature_dataset.user_feat.columns),
            ["user_id", "age", "gender", "occupation"],
        )
        self.assertEqual(
            list(feature_dataset.item_feat.columns),
            ["item_id", "release_year"],
        )

    def test_extract_feature_data_aligns_to_internal_ids(self):
        _, dataset, _, _, _ = _require_ml100k(feature_aware=True)
        feature_data = extract_feature_data(dataset, "ml-100k")

        self.assertEqual(
            [spec["name"] for spec in feature_data["user"]["schema"]],
            ["age", "gender", "occupation"],
        )
        self.assertEqual(
            [spec["name"] for spec in feature_data["item"]["schema"]],
            ["release_year"],
        )

        age_tensor = feature_data["user"]["tensors"]["age"]
        release_year_tensor = feature_data["item"]["tensors"]["release_year"]

        self.assertEqual(age_tensor.shape[0], dataset.num(dataset.uid_field))
        self.assertEqual(release_year_tensor.shape[0], dataset.num(dataset.iid_field))
        self.assertEqual(age_tensor[0].item(), 0)
        self.assertEqual(release_year_tensor[0].item(), 0)

        user_row = 10
        item_row = 10
        user_id = int(dataset.user_feat["user_id"][user_row])
        item_id = int(dataset.item_feat["item_id"][item_row])

        self.assertEqual(
            age_tensor[user_id].item(),
            int(dataset.user_feat["age"][user_row]),
        )
        self.assertEqual(
            release_year_tensor[item_id].item(),
            int(dataset.item_feat["release_year"][item_row]),
        )

    def test_ml100k_feature_aware_integration_smoke(self):
        _, dataset, train_data, _, _ = _require_ml100k(feature_aware=True)
        feature_data = extract_feature_data(dataset, "ml-100k")

        model = TwoTowerModel(
            num_users=dataset.num(dataset.uid_field),
            num_items=dataset.num(dataset.iid_field),
            embedding_size=8,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            user_feature_schema=feature_data["user"]["schema"],
            user_feature_tensors=feature_data["user"]["tensors"],
            item_feature_schema=feature_data["item"]["schema"],
            item_feature_tensors=feature_data["item"]["tensors"],
        )

        batch = next(iter(train_data))
        user_ids = batch["user_id"][:4]
        pos_item_ids = batch["item_id"][:4]
        neg_item_ids = torch.tensor([[5, 6], [7, 8], [9, 10], [11, 12]])

        loss = model.compute_loss(user_ids, pos_item_ids, neg_item_ids)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.get_user_embedding(user_ids).shape, (4, 8))
        self.assertEqual(model.get_item_embedding(pos_item_ids).shape, (4, 8))


class FeatureAwareModelTests(unittest.TestCase):
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

    def test_feature_aware_mode_preserves_shapes_and_loss_behavior(self):
        model = TwoTowerModel(
            num_users=5,
            num_items=6,
            embedding_size=8,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            user_feature_schema=[
                {"name": "age", "type": "token", "num_embeddings": 5},
                {"name": "gender", "type": "token", "num_embeddings": 4},
            ],
            user_feature_tensors={
                "age": torch.tensor([1, 2, 3, 2, 1]),
                "gender": torch.tensor([1, 2, 1, 2, 1]),
            },
            item_feature_schema=[
                {"name": "release_year", "type": "token", "num_embeddings": 6}
            ],
            item_feature_tensors={
                "release_year": torch.tensor([1, 2, 3, 4, 5, 1])
            },
        )
        user_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        pos_item_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        neg_item_ids = torch.tensor([[4, 5], [0, 5], [1, 4]], dtype=torch.long)

        loss = model.compute_loss(user_ids, pos_item_ids, neg_item_ids)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.get_user_embedding(user_ids).shape, (3, 8))
        self.assertEqual(model.get_item_embedding(pos_item_ids).shape, (3, 8))
        self.assertEqual(model.get_item_embedding(neg_item_ids).shape, (3, 2, 8))
        self.assertEqual(model.predict(user_ids).shape, (3, 6))

    def test_token_sequence_pooling_ignores_padding_tokens(self):
        tower = Tower(
            input_dim=3,
            embedding_size=2,
            hidden_size=4,
            num_layers=0,
            dropout=0.0,
            feature_schema=[
                {
                    "name": "movie_title",
                    "type": "token_seq",
                    "num_embeddings": 6,
                    "max_length": 3,
                }
            ],
            feature_tensors={
                "movie_title": torch.tensor(
                    [
                        [0, 0, 0],
                        [1, 2, 0],
                        [3, 4, 5],
                    ],
                    dtype=torch.long,
                )
            },
        )
        with torch.no_grad():
            tower.side_embeddings["movie_title"].weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [1.0, 1.0],
                        [3.0, 3.0],
                        [2.0, 0.0],
                        [4.0, 0.0],
                        [6.0, 0.0],
                    ]
                )
            )

        pooled = tower._encode_side_feature(
            "movie_title", torch.tensor([1, 2], dtype=torch.long)
        )
        expected = torch.tensor([[2.0, 2.0], [4.0, 0.0]])

        self.assertTrue(torch.allclose(pooled, expected))

    def test_hard_sampler_works_with_feature_aware_model(self):
        model = TwoTowerModel(
            num_users=4,
            num_items=8,
            embedding_size=8,
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            user_feature_schema=[
                {"name": "age", "type": "token", "num_embeddings": 5}
            ],
            user_feature_tensors={"age": torch.tensor([1, 2, 3, 4])},
            item_feature_schema=[
                {"name": "release_year", "type": "token", "num_embeddings": 6}
            ],
            item_feature_tensors={
                "release_year": torch.tensor([1, 2, 3, 4, 5, 1, 2, 3])
            },
        )
        user_item_dict = {
            0: {1, 2},
            1: {2, 3},
            2: {4},
            3: {5},
        }
        sampler = get_sampler(
            "hard",
            num_items=8,
            num_neg_samples=2,
            user_item_dict=user_item_dict,
            model=model,
            candidate_pool_size=4,
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
