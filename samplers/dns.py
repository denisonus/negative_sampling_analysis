"""Softmax-based dynamic negative sampling."""

import torch
from typing import Dict, Optional, Set

from .base import NegativeSampler, Device
from .hard import EmbeddingModel


class DNSNegativeSampler(NegativeSampler):
    """Score a candidate pool with the model and sample via softmax."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        model: Optional[EmbeddingModel] = None,
        device: Device = "cpu",
        candidate_pool_size: int = 100,
        temperature: float = 0.1,
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "dns"
        self.model = model
        self.candidate_pool_size = candidate_pool_size
        self.temperature = temperature

    def set_model(self, model: EmbeddingModel) -> None:
        """Set the model for computing embeddings."""
        self.model = model

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model must be set before sampling DNS negatives")

        batch_size = user_ids.size(0)

        all_candidates = self._sample_candidate_pools_batch(
            user_ids,
            self.candidate_pool_size,
            oversample=self.candidate_pool_size * 3,
        )

        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_ids)
            all_cand_emb = self.model.get_item_embedding(all_candidates.view(-1)).view(
                batch_size, self.candidate_pool_size, -1
            )
            scores = torch.bmm(
                user_emb.unsqueeze(1), all_cand_emb.transpose(1, 2)
            ).squeeze(1)

            probs = torch.softmax(scores / self.temperature, dim=1)
            sampled_indices = torch.multinomial(
                probs, self.num_neg_samples, replacement=False
            )
            neg_items = torch.gather(all_candidates, 1, sampled_indices)

        return neg_items
