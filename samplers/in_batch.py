"""In-batch negative sampling.

Uses other positive items in the same mini-batch as negatives,
a computationally efficient strategy widely used in two-tower models.

Reference:
    Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus
    Item Recommendations" (RecSys 2019).

    Yang et al., "Mixed Negative Sampling for Learning Two-tower Neural
    Networks in Recommendations" (WWW 2020).
"""

import torch
from typing import Set, Dict

from .base import NegativeSampler, Device


class InBatchNegativeSampler(NegativeSampler):
    """In-batch negative sampling — uses other items in batch as negatives.

    Note: The actual in-batch negative logic is implemented in InBatchTrainer,
    where all positive items in the batch serve as negatives for other users.
    This sampler satisfies the sampler interface but returns empty tensors
    since sampling is handled during the forward pass.

    Reference:
        Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus
        Item Recommendations" (RecSys 2019).
    """

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
        """Return empty tensor - in-batch negatives are handled by the trainer."""
        batch_size = user_ids.size(0)
        return torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
