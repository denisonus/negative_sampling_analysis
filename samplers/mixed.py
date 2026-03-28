"""Mixed negative sampling."""

import numpy as np
import torch
from typing import Set, Dict, Optional

from .base import NegativeSampler, Device
from .uniform import UniformNegativeSampler
from .hard import HardNegativeSampler, EmbeddingModel


class MixedNegativeSampler(NegativeSampler):
    """Combine random and hard negatives with a fixed ratio."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        model: Optional[EmbeddingModel] = None,
        device: Device = "cpu",
        hard_ratio: float = 0.5,
        candidate_pool_size: int = 100,
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "mixed"
        self.hard_ratio = hard_ratio

        self.num_hard = int(num_neg_samples * hard_ratio)
        self.num_random = num_neg_samples - self.num_hard

        self.uniform_sampler = UniformNegativeSampler(
            num_items, self.num_random, user_item_dict, device
        )
        self.hard_sampler = HardNegativeSampler(
            num_items, self.num_hard, user_item_dict, model, device, candidate_pool_size
        )

    def set_model(self, model: EmbeddingModel) -> None:
        """Set the model for hard negative sampling."""
        self.hard_sampler.set_model(model)

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_size = user_ids.size(0)

        # Sample random negatives
        if self.num_random > 0:
            random_negs = self.uniform_sampler.sample(user_ids, pos_item_ids)
        else:
            random_negs = torch.empty(
                batch_size, 0, dtype=torch.long, device=self.device
            )

        # Sample hard negatives
        if self.num_hard > 0 and self.hard_sampler.model is not None:
            hard_negs = self.hard_sampler.sample(user_ids, pos_item_ids)
        else:
            hard_negs = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

        combined = torch.cat([hard_negs, random_negs], dim=1)
        if combined.size(1) == 0:
            return combined

        deduped = np.zeros((batch_size, self.num_neg_samples), dtype=np.int64)
        user_ids_np = user_ids.cpu().numpy()
        combined_np = combined.cpu().numpy()

        for i in range(batch_size):
            positives = self._get_positives(user_ids_np[i])
            deduped[i] = self._sample_unique_valid_items(
                combined_np[i], positives, self.num_neg_samples
            )

        return torch.from_numpy(deduped).to(self.device)
