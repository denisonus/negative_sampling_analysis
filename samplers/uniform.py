"""Uniform random negative sampling."""

import torch
import numpy as np
from typing import Set, Dict

from .base import NegativeSampler, Device


class UniformNegativeSampler(NegativeSampler):
    """Sample negatives uniformly from items the user has not seen."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        device: Device = "cpu",
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "uniform"

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)
        oversample = max(self.num_neg_samples * 3, self.num_neg_samples + 50)
        candidates = np.random.randint(0, self.num_items, size=(batch_size, oversample))

        neg_items = np.zeros((batch_size, self.num_neg_samples), dtype=np.int64)
        user_ids_np = user_ids.cpu().numpy()

        for i in range(batch_size):
            positives = self._get_positives(user_ids_np[i])
            neg_items[i] = self._sample_unique_valid_items(
                candidates[i], positives, self.num_neg_samples
            )

        return torch.from_numpy(neg_items).to(self.device)
