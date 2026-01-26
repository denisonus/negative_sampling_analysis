"""Curriculum Learning negative sampling."""

import torch
import numpy as np
from typing import Set, Dict, Optional

from .base import NegativeSampler, Device
from .uniform import UniformNegativeSampler
from .hard import EmbeddingModel


class CurriculumNegativeSampler(NegativeSampler):
    """Curriculum Learning - gradually increase negative difficulty during training.
    
    Starts with easy (random) negatives and progressively introduces harder negatives
    as training progresses, following curriculum learning principles.
    
    Reference: Bengio et al., "Curriculum Learning"
    """
    
    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        model: Optional[EmbeddingModel] = None,
        device: Device = 'cpu',
        candidate_pool_size: int = 100,
        start_hard_ratio: float = 0.0,
        end_hard_ratio: float = 0.8,
        warmup_epochs: int = 10
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = 'curriculum'
        self.model = model
        self.candidate_pool_size = candidate_pool_size
        self.start_hard_ratio = start_hard_ratio
        self.end_hard_ratio = end_hard_ratio
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # Sub-sampler for random negatives
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
        return self.start_hard_ratio + progress * (self.end_hard_ratio - self.start_hard_ratio)
    
    def sample(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        hard_ratio = self._get_current_hard_ratio()
        num_hard = int(self.num_neg_samples * hard_ratio)
        num_random = self.num_neg_samples - num_hard
        
        batch_size = user_ids.size(0)
        
        # Sample random negatives
        if num_random > 0:
            self._uniform_sampler.num_neg_samples = num_random
            random_negs = self._uniform_sampler.sample(user_ids, pos_item_ids)
        else:
            random_negs = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        
        # Sample hard negatives
        if num_hard > 0 and self.model is not None:
            hard_negs = self._sample_hard(user_ids, num_hard)
        else:
            hard_negs = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        
        return torch.cat([hard_negs, random_negs], dim=1)
    
    def _sample_hard(self, user_ids: torch.Tensor, num_hard: int) -> torch.Tensor:
        """Sample hard negatives using model scores."""
        batch_size = user_ids.size(0)
        neg_items = torch.zeros(batch_size, num_hard, dtype=torch.long)
        
        # Model is guaranteed to be not None here (checked in sample())
        assert self.model is not None
        
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(user_ids)
            
            for i in range(batch_size):
                user_id = user_ids[i].item()
                positives = self._get_positives(user_id)
                
                # Generate candidate pool
                candidates = []
                while len(candidates) < self.candidate_pool_size:
                    c = np.random.randint(0, self.num_items)
                    if c not in positives:
                        candidates.append(c)
                
                candidates_tensor = torch.tensor(candidates, device=self.device)
                cand_emb = self.model.get_item_embedding(candidates_tensor)
                scores = torch.matmul(user_emb[i:i+1], cand_emb.t()).squeeze(0)
                
                k = min(num_hard, len(candidates))
                _, top_indices = torch.topk(scores, k)
                neg_items[i, :k] = candidates_tensor[top_indices]
        
        return neg_items.to(self.device)
