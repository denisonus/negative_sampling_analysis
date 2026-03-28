"""Shared sampler interfaces."""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Set, Dict, Union, Any, Optional
from dataclasses import dataclass

Device = Union[str, torch.device]


@dataclass
class SamplingResult:
    """Negative samples with optional per-sample log probabilities."""

    neg_items: torch.Tensor
    log_probs: Optional[torch.Tensor] = None


class NegativeSampler(ABC):
    """Base interface for negative samplers."""

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        device: Device = "cpu",
    ):
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        self.user_item_dict = user_item_dict
        self.device = device
        self.name = "base"
        self.return_probs = False

    @abstractmethod
    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> Union[torch.Tensor, SamplingResult]:
        """Sample negatives for the provided users and positive items."""
        pass

    def set_model(self, model: Any) -> None:
        """Optional hook for samplers that need model access."""
        return None

    def set_epoch(self, epoch: int) -> None:
        """Optional hook for samplers that need epoch context."""
        return None

    def _get_positives(self, user_id: Any) -> Set[int]:
        """Get positive items for a user."""
        return self.user_item_dict.get(int(user_id), set())

    def _sample_unique_valid_items(
        self, candidates: np.ndarray, positives: Set[int], count: int
    ) -> np.ndarray:
        """Return up to ``count`` negatives, preferring unique items.

        Duplicates are removed while preserving order. If there are not enough
        unique valid items available for a very dense user, the remainder is
        filled with valid items allowing duplicates to avoid infinite loops.
        """
        if count <= 0:
            return np.empty(0, dtype=np.int64)

        chosen = []
        chosen_set = set()

        for item in candidates:
            item = int(item)
            if item in positives or item in chosen_set:
                continue
            chosen.append(item)
            chosen_set.add(item)
            if len(chosen) == count:
                return np.array(chosen, dtype=np.int64)

        max_unique = max(self.num_items - len(positives), 0)
        target_unique = min(count, max_unique)

        while len(chosen) < target_unique:
            item = int(np.random.randint(0, self.num_items))
            if item in positives or item in chosen_set:
                continue
            chosen.append(item)
            chosen_set.add(item)

        while len(chosen) < count:
            item = int(np.random.randint(0, self.num_items))
            if item not in positives:
                chosen.append(item)

        return np.array(chosen, dtype=np.int64)
