"""Hard negative mining."""

import torch
import numpy as np
from typing import Set, Dict, Optional, Protocol, runtime_checkable

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

        all_candidates, valid_counts = self._sample_candidate_pools_batch(user_ids)

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

    def _sample_candidate_pools_batch(self, user_ids: torch.Tensor) -> tuple:
        """Sample candidate pools for all users in batch."""
        batch_size = user_ids.size(0)
        oversample = self.candidate_pool_size + 50
        candidates = np.random.randint(0, self.num_items, size=(batch_size, oversample))

        result = np.zeros((batch_size, self.candidate_pool_size), dtype=np.int64)
        valid_counts = np.zeros(batch_size, dtype=np.int64)
        user_ids_np = user_ids.cpu().numpy()

        for i in range(batch_size):
            positives = self._get_positives(user_ids_np[i])
            row = candidates[i]
            mask = np.isin(row, list(positives), invert=True)
            valid = row[mask]
            count = min(len(valid), self.candidate_pool_size)
            result[i, :count] = valid[:count]
            valid_counts[i] = count
            if count < self.candidate_pool_size:
                idx = count
                while idx < self.candidate_pool_size:
                    c = np.random.randint(0, self.num_items)
                    if c not in positives:
                        result[i, idx] = c
                        idx += 1

        return torch.from_numpy(result).to(self.device), valid_counts
