"""In-batch + uniform mixed negative sampling."""

import numpy as np
import torch
from typing import Dict, Optional, Set

from .base import NegativeSampler, Device


class MixedInBatchUniformNegativeSampler(NegativeSampler):
    """Combine in-batch negatives with shared uniform negatives."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        device: Device = "cpu",
        index_batch_size: int = 1024,
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "mixed_in_batch_uniform"
        self.index_batch_size = max(int(index_batch_size), 0)

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return an empty tensor because the trainer builds mixed negatives."""
        batch_size = user_ids.size(0)
        return torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

    def sample_shared_uniform_items(
        self,
        exclude_item_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample a shared uniform item set from the corpus for the whole batch."""
        if self.index_batch_size <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        excluded = set()
        if exclude_item_ids is not None:
            excluded = {int(item) for item in exclude_item_ids.detach().cpu().tolist()}

        available = self.num_items - len(excluded)
        if available <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        count = min(self.index_batch_size, available)
        candidates = np.arange(self.num_items, dtype=np.int64)
        if excluded:
            mask = np.ones(self.num_items, dtype=bool)
            mask[list(excluded)] = False
            candidates = candidates[mask]

        replace = count > len(candidates)
        sampled = np.random.choice(candidates, size=count, replace=replace)
        return torch.from_numpy(sampled.astype(np.int64)).to(self.device)
