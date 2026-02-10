"""Dynamic Negative Sampling (DNS)."""

import torch
import numpy as np
from typing import Set, Dict, Optional

from .base import NegativeSampler, Device
from .hard import EmbeddingModel


class DNSNegativeSampler(NegativeSampler):
    """Dynamic Negative Sampling (DNS) - online hard negative mining with softmax sampling.

    Instead of taking the hardest negatives (which can be noisy), DNS samples
    negatives proportionally to their scores using softmax over a candidate pool.

    Reference: Zhang et al., "Optimizing Top-N Collaborative Filtering via Dynamic Negative Sampling"
    """

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        model: Optional[EmbeddingModel] = None,
        device: Device = "cpu",
        candidate_pool_size: int = 100,
        temperature: float = 1.0,
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
        neg_items = torch.zeros(
            batch_size, self.num_neg_samples, dtype=torch.long, device=self.device
        )

        # Generate candidate pools for all users at once
        all_candidates = self._sample_candidate_pools_batch(user_ids)

        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_ids)
            # Get embeddings for all candidates at once
            all_cand_emb = self.model.get_item_embedding(all_candidates.view(-1)).view(
                batch_size, self.candidate_pool_size, -1
            )
            # Compute scores: (batch, 1, dim) @ (batch, dim, pool) -> (batch, 1, pool)
            scores = torch.bmm(
                user_emb.unsqueeze(1), all_cand_emb.transpose(1, 2)
            ).squeeze(1)

            # Softmax sampling for each user
            probs = torch.softmax(scores / self.temperature, dim=1)
            sampled_indices = torch.multinomial(
                probs, self.num_neg_samples, replacement=False
            )
            neg_items = torch.gather(all_candidates, 1, sampled_indices)

        return neg_items

    def _sample_candidate_pools_batch(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Sample candidate pools for all users in batch."""
        batch_size = user_ids.size(0)
        oversample = self.candidate_pool_size + 50
        candidates = np.random.randint(0, self.num_items, size=(batch_size, oversample))

        result = np.zeros((batch_size, self.candidate_pool_size), dtype=np.int64)
        user_ids_np = user_ids.cpu().numpy()

        for i in range(batch_size):
            positives = self._get_positives(user_ids_np[i])
            row = candidates[i]
            mask = np.isin(row, list(positives), invert=True)
            valid = row[mask]
            count = min(len(valid), self.candidate_pool_size)
            result[i, :count] = valid[:count]
            # Fill remaining with random if needed
            if count < self.candidate_pool_size:
                idx = count
                while idx < self.candidate_pool_size:
                    c = np.random.randint(0, self.num_items)
                    if c not in positives:
                        result[i, idx] = c
                        idx += 1

        return torch.from_numpy(result).to(self.device)
