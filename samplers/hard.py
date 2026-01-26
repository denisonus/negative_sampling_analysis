"""Hard negative mining sampler."""

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
    """Hard negative mining - samples negatives with high model scores.
    
    Generates a candidate pool of random negatives, then selects
    the ones with highest predicted scores as hard negatives.
    """
    
    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        model: Optional[EmbeddingModel] = None,
        device: Device = 'cpu',
        candidate_pool_size: int = 100
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = 'hard'
        self.model = model
        self.candidate_pool_size = candidate_pool_size
    
    def set_model(self, model: EmbeddingModel) -> None:
        """Set the model for computing embeddings."""
        self.model = model
    
    def sample(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model must be set before sampling hard negatives")
        
        batch_size = user_ids.size(0)
        neg_items = torch.zeros(batch_size, self.num_neg_samples, dtype=torch.long)
        
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_ids)
            
            for i in range(batch_size):
                user_id = user_ids[i].item()
                positives = self._get_positives(user_id)
                
                # Generate candidate pool
                candidates = self._sample_candidate_pool(positives)
                candidates_tensor = torch.tensor(candidates, device=self.device)
                
                # Score candidates and select hardest
                cand_emb = self.model.get_item_embedding(candidates_tensor)
                scores = torch.matmul(user_emb[i:i+1], cand_emb.t()).squeeze(0)
                
                k = min(self.num_neg_samples, len(candidates))
                _, top_indices = torch.topk(scores, k)
                neg_items[i, :k] = candidates_tensor[top_indices]
        
        return neg_items.to(self.device)
    
    def _sample_candidate_pool(self, positives: Set[int]) -> list:
        """Sample a pool of candidate negatives."""
        candidates = []
        while len(candidates) < self.candidate_pool_size:
            c = np.random.randint(0, self.num_items)
            if c not in positives:
                candidates.append(c)
        return candidates
