"""Hard negative mining."""

import torch
from typing import Dict, Optional, Protocol, Set, runtime_checkable

from .base import NegativeSampler, Device


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for models that provide embedding methods."""

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor: ...
    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor: ...


class HardNegativeSampler(NegativeSampler):
    """Sample a candidate pool and keep the highest-scoring negatives."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        model: Optional[EmbeddingModel] = None,
        device: Device = "cpu",
        candidate_pool_size: int = 100,
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "hard"
        self.model = model
        self.candidate_pool_size = candidate_pool_size

    def set_model(self, model: EmbeddingModel) -> None:
        """Set the model for computing embeddings."""
        self.model = model

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model must be set before sampling hard negatives")

        batch_size = user_ids.size(0)
        neg_items = torch.zeros(
            batch_size, self.num_neg_samples, dtype=torch.long, device=self.device
        )

        all_candidates = self._sample_candidate_pools_batch(
            user_ids, self.candidate_pool_size
        )

        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_ids)
            all_cand_emb = self.model.get_item_embedding(all_candidates.view(-1)).view(
                batch_size, self.candidate_pool_size, -1
            )
            scores = torch.bmm(
                user_emb.unsqueeze(1), all_cand_emb.transpose(1, 2)
            ).squeeze(1)

            k = min(self.num_neg_samples, self.candidate_pool_size)
            _, top_indices = torch.topk(scores, k, dim=1)
            neg_items[:, :k] = torch.gather(all_candidates, 1, top_indices)

        return neg_items
