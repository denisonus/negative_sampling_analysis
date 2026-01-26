"""Uniform random negative sampling."""

import torch
import numpy as np
from typing import Set, Dict

from .base import NegativeSampler, Device


class UniformNegativeSampler(NegativeSampler):
    """Uniform random negative sampling.
    
    Samples negatives uniformly at random from all items,
    excluding items the user has already interacted with.
    """
    
    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        device: Device = 'cpu'
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = 'uniform'
    
    def sample(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        batch_size = user_ids.size(0)
        neg_items = torch.zeros(batch_size, self.num_neg_samples, dtype=torch.long)
        
        for i in range(batch_size):
            user_id = user_ids[i].item()
            positives = self._get_positives(user_id)
            
            neg_samples = []
            while len(neg_samples) < self.num_neg_samples:
                candidates = np.random.randint(0, self.num_items, size=self.num_neg_samples * 2)
                for c in candidates:
                    if c not in positives:
                        neg_samples.append(c)
                        if len(neg_samples) >= self.num_neg_samples:
                            break
            
            neg_items[i] = torch.tensor(neg_samples[:self.num_neg_samples])
        
        return neg_items.to(self.device)
