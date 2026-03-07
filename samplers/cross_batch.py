"""Cross-Batch Negative Sampling (CBNS).

Extends in-batch negatives by maintaining a FIFO queue of item embeddings
from recent mini-batches, effectively decoupling the number of negatives
from the batch size.

Reference:
    Wang et al., "Cross-Batch Negative Sampling for Training Two-Tower
    Recommenders" (2021).
"""

import torch
from typing import Set, Dict

from .base import NegativeSampler, Device


class CrossBatchNegativeSampler(NegativeSampler):
    """Cross-Batch Negative Sampling — uses queued embeddings as extra negatives.

    Note: The actual cross-batch logic is implemented in CrossBatchTrainer,
    where a FIFO queue of item embeddings from recent batches supplements
    in-batch negatives. This sampler satisfies the sampler interface but
    returns empty tensors since sampling is handled during the forward pass.

    Reference:
        Wang et al., "Cross-Batch Negative Sampling for Training Two-Tower
        Recommenders" (2021).
    """

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        device: Device = "cpu",
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "cross_batch"

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return empty tensor - cross-batch negatives are handled by the trainer."""
        batch_size = user_ids.size(0)
        return torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
