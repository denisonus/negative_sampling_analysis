"""Curriculum-based negative sampling."""

import torch
import numpy as np
from typing import Set, Dict, Optional

from .base import NegativeSampler, Device
from .uniform import UniformNegativeSampler
from .hard import EmbeddingModel


class CurriculumNegativeSampler(NegativeSampler):
    """Increase the share of hard negatives over the warmup period."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        model: Optional[EmbeddingModel] = None,
        device: Device = "cpu",
        candidate_pool_size: int = 100,
        start_hard_ratio: float = 0.0,
        end_hard_ratio: float = 0.8,
        warmup_epochs: int = 10,
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "curriculum"
        self.model = model
        self.candidate_pool_size = candidate_pool_size
        self.start_hard_ratio = start_hard_ratio
        self.end_hard_ratio = end_hard_ratio
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self._uniform_sampler = UniformNegativeSampler(
            num_items, num_neg_samples, user_item_dict, device
        )

    def set_model(self, model: EmbeddingModel) -> None:
        """Set the model for hard negative sampling."""
        self.model = model

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch

    def _get_current_hard_ratio(self) -> float:
        """Calculate current hard ratio based on training progress."""
        if self.current_epoch >= self.warmup_epochs:
            return self.end_hard_ratio
        progress = self.current_epoch / self.warmup_epochs
        return self.start_hard_ratio + progress * (
            self.end_hard_ratio - self.start_hard_ratio
        )

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        hard_ratio = self._get_current_hard_ratio()
        num_hard = int(self.num_neg_samples * hard_ratio)
        num_random = self.num_neg_samples - num_hard

        batch_size = user_ids.size(0)

        # Sample random negatives
        if num_random > 0:
            self._uniform_sampler.num_neg_samples = num_random
            random_negs = self._uniform_sampler.sample(user_ids, pos_item_ids)
        else:
            random_negs = torch.empty(
                batch_size, 0, dtype=torch.long, device=self.device
            )

        # Sample hard negatives
        if num_hard > 0 and self.model is not None:
            hard_negs = self._sample_hard(user_ids, num_hard)
        else:
            hard_negs = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

        return torch.cat([hard_negs, random_negs], dim=1)

    def _sample_hard(self, user_ids: torch.Tensor, num_hard: int) -> torch.Tensor:
        """Sample hard negatives using model scores."""
        batch_size = user_ids.size(0)
        neg_items = torch.zeros(
            batch_size, num_hard, dtype=torch.long, device=self.device
        )

        assert self.model is not None

        all_candidates = self._sample_candidate_pools_batch(user_ids)

        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_ids)
            all_cand_emb = self.model.get_item_embedding(all_candidates.view(-1)).view(
                batch_size, self.candidate_pool_size, -1
            )
            scores = torch.bmm(
                user_emb.unsqueeze(1), all_cand_emb.transpose(1, 2)
            ).squeeze(1)

            k = min(num_hard, self.candidate_pool_size)
            _, top_indices = torch.topk(scores, k, dim=1)
            neg_items[:, :k] = torch.gather(all_candidates, 1, top_indices)

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
            if count < self.candidate_pool_size:
                idx = count
                while idx < self.candidate_pool_size:
                    c = np.random.randint(0, self.num_items)
                    if c not in positives:
                        result[i, idx] = c
                        idx += 1

        return torch.from_numpy(result).to(self.device)
