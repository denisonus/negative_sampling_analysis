"""Popularity-based negative sampling."""

import torch
import numpy as np
from typing import Set, Dict, List, Union

from .base import NegativeSampler, Device


class PopularityNegativeSampler(NegativeSampler):
    """Popularity-based negative sampling.
    
    Samples negatives proportionally to item popularity (with smoothing).
    More popular items are more likely to be sampled as negatives,
    which provides harder negatives without requiring model inference.
    """
    
    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        item_popularity: Union[List[float], np.ndarray],
        device: Device = 'cpu',
        smoothing: float = 0.75
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = 'popularity'
        
        # Apply smoothing to reduce bias towards very popular items
        popularity = np.array(item_popularity, dtype=np.float64)
        popularity = np.power(popularity + 1e-10, smoothing)  # Add small epsilon to avoid zero
        self.sampling_probs = popularity / popularity.sum()
    
    def sample(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        batch_size = user_ids.size(0)
        neg_items = torch.zeros(batch_size, self.num_neg_samples, dtype=torch.long)
        
        for i in range(batch_size):
            user_id = user_ids[i].item()
            positives = self._get_positives(user_id)
            
            neg_samples = []
            while len(neg_samples) < self.num_neg_samples:
                candidates = np.random.choice(
                    self.num_items, 
                    size=self.num_neg_samples * 2,
                    p=self.sampling_probs, 
                    replace=True
                )
                for c in candidates:
                    if c not in positives:
                        neg_samples.append(c)
                        if len(neg_samples) >= self.num_neg_samples:
                            break
            
            neg_items[i] = torch.tensor(neg_samples[:self.num_neg_samples])
        
        return neg_items.to(self.device)
