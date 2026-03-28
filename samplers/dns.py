"""Softmax-based dynamic negative sampling."""

import torch
import numpy as np
from typing import Set, Dict, Optional

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

        self._positives_array: Dict[int, np.ndarray] = {}
        for user_id, items in user_item_dict.items():
            self._positives_array[user_id] = np.array(list(items), dtype=np.int64)

    def set_model(self, model: EmbeddingModel) -> None:
        """Set the model for computing embeddings."""
        self.model = model

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model must be set before sampling DNS negatives")

        batch_size = user_ids.size(0)

        all_candidates = self._sample_candidate_pools_batch(user_ids)

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

    def _sample_candidate_pools_batch(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Sample candidate pools for all users in batch."""
        batch_size = user_ids.size(0)
        user_ids_np = user_ids.cpu().numpy()

        oversample_factor = 3
        total_needed = self.candidate_pool_size * oversample_factor

        all_candidates = np.random.randint(
            0, self.num_items, size=(batch_size, total_needed), dtype=np.int64
        )

        result = np.zeros((batch_size, self.candidate_pool_size), dtype=np.int64)

        for i in range(batch_size):
            user_id = user_ids_np[i]
            positives = self._positives_array.get(user_id)
            positives_set = set() if positives is None else set(positives.tolist())
            result[i] = self._sample_unique_valid_items(
                all_candidates[i], positives_set, self.candidate_pool_size
            )

        return torch.from_numpy(result).to(self.device)
