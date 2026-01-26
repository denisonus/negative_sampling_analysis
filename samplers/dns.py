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
        device: Device = 'cpu',
        candidate_pool_size: int = 100,
        temperature: float = 1.0
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = 'dns'
        self.model = model
        self.candidate_pool_size = candidate_pool_size
        self.temperature = temperature
    
    def set_model(self, model: EmbeddingModel) -> None:
        """Set the model for computing embeddings."""
        self.model = model
    
    def sample(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model must be set before sampling DNS negatives")
        
        batch_size = user_ids.size(0)
        neg_items = torch.zeros(batch_size, self.num_neg_samples, dtype=torch.long)
        
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_ids)
            
            for i in range(batch_size):
                user_id = user_ids[i].item()
                positives = self._get_positives(user_id)
                
                # Sample candidate pool uniformly
                candidates = self._sample_candidate_pool(positives)
                candidates_tensor = torch.tensor(candidates, device=self.device)
                
                # Compute scores
                cand_emb = self.model.get_item_embedding(candidates_tensor)
                scores = torch.matmul(user_emb[i:i+1], cand_emb.t()).squeeze(0)
                
                # Softmax sampling instead of top-k
                probs = torch.softmax(scores / self.temperature, dim=0)
                sampled_indices = torch.multinomial(probs, self.num_neg_samples, replacement=False)
                neg_items[i] = candidates_tensor[sampled_indices]
        
        return neg_items.to(self.device)
    
    def _sample_candidate_pool(self, positives: Set[int]) -> list:
        """Sample a pool of candidate negatives."""
        candidates = []
        while len(candidates) < self.candidate_pool_size:
            c = np.random.randint(0, self.num_items)
            if c not in positives:
                candidates.append(c)
        return candidates
