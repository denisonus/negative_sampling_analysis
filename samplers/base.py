"""Shared sampler interfaces."""

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
