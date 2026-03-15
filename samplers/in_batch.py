"""In-batch negative sampling."""

import torch
from typing import Set, Dict

from .base import NegativeSampler, Device


class InBatchNegativeSampler(NegativeSampler):
    """Interface placeholder for trainer-managed in-batch negatives."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        device: Device = "cpu",
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "in_batch"

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return an empty tensor because the trainer builds in-batch negatives."""
        batch_size = user_ids.size(0)
        return torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
