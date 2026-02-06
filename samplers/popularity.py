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
        # Over-sample to account for filtering out positives
        oversample = max(self.num_neg_samples * 3, self.num_neg_samples + 50)
        candidates = np.random.choice(
            self.num_items, 
            size=(batch_size, oversample),
            p=self.sampling_probs, 
            replace=True
        )
        
        neg_items = np.zeros((batch_size, self.num_neg_samples), dtype=np.int64)
        user_ids_np = user_ids.cpu().numpy()
        
        for i in range(batch_size):
            positives = self._get_positives(user_ids_np[i])
            row = candidates[i]
            # Vectorized filtering
            mask = np.isin(row, list(positives), invert=True)
            valid = row[mask]
            
            if len(valid) >= self.num_neg_samples:
                neg_items[i] = valid[:self.num_neg_samples]
            else:
                # Rare case: need more samples
                neg_items[i, :len(valid)] = valid
                idx = len(valid)
                while idx < self.num_neg_samples:
                    c = np.random.choice(self.num_items, p=self.sampling_probs)
                    if c not in positives:
                        neg_items[i, idx] = c
                        idx += 1
        
        return torch.from_numpy(neg_items).to(self.device)
