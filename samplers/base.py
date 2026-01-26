"""Base class for negative samplers."""

import torch
from abc import ABC, abstractmethod
from typing import Set, Dict, Union, Any

# Type alias for device
Device = Union[str, torch.device]


class NegativeSampler(ABC):
    """Base class for negative samplers."""
    
    def __init__(
        self, 
        num_items: int, 
        num_neg_samples: int, 
        user_item_dict: Dict[int, Set[int]], 
        device: Device = 'cpu'
    ):
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        self.user_item_dict = user_item_dict
        self.device = device
        self.name = 'base'
    
    @abstractmethod
    def sample(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        """Sample negative items for given users.
        
        Args:
            user_ids: Tensor of user IDs (batch_size,)
            pos_item_ids: Tensor of positive item IDs (batch_size,)
            
        Returns:
            Tensor of negative item IDs (batch_size, num_neg_samples)
        """
        pass
    
    def _get_positives(self, user_id: Any) -> Set[int]:
        """Get positive items for a user."""
        return self.user_item_dict.get(int(user_id), set())
