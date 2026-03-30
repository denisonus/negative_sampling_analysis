"""Factory function for creating negative samplers."""

import numpy as np
from typing import Set, Dict, Optional, List, Union

from .base import NegativeSampler, Device
from .uniform import UniformNegativeSampler
from .popularity import PopularityNegativeSampler
from .hard import HardNegativeSampler, EmbeddingModel
from .mixed import MixedHardUniformNegativeSampler
from .in_batch import InBatchNegativeSampler
from .mixed_in_batch_uniform import MixedInBatchUniformNegativeSampler
from .dns import DNSNegativeSampler
from .curriculum import CurriculumNegativeSampler
from .debiased import DebiasedNegativeSampler


def get_sampler(
    strategy: str,
    num_items: int,
    num_neg_samples: int,
    user_item_dict: Dict[int, Set[int]],
    item_popularity: Optional[Union[List[float], np.ndarray]] = None,
    model: Optional[EmbeddingModel] = None,
    device: Device = "cpu",
    **kwargs,
) -> NegativeSampler:

    strategy = strategy.lower()

    if strategy == "uniform":
        return UniformNegativeSampler(
            num_items, num_neg_samples, user_item_dict, device
        )

    elif strategy == "popularity":
        if item_popularity is None:
            item_popularity = np.ones(num_items)
        return PopularityNegativeSampler(
            num_items,
            num_neg_samples,
            user_item_dict,
            item_popularity,
            device,
            smoothing=kwargs.get("smoothing", 0.75),
            logq_correction=kwargs.get("logq_correction", True),
        )

    elif strategy == "hard":
        return HardNegativeSampler(
            num_items,
            num_neg_samples,
            user_item_dict,
            model,
            device,
            candidate_pool_size=kwargs.get("candidate_pool_size", 100),
        )

    elif strategy in {"mixed_hard_uniform", "mixed"}:
        return MixedHardUniformNegativeSampler(
            num_items,
            num_neg_samples,
            user_item_dict,
            model,
            device,
            hard_ratio=kwargs.get("hard_ratio", 0.5),
            candidate_pool_size=kwargs.get("candidate_pool_size", 100),
        )

    elif strategy == "in_batch":
        return InBatchNegativeSampler(
            num_items, num_neg_samples, user_item_dict, device
        )

    elif strategy in {"mixed_in_batch_uniform", "mns"}:
        return MixedInBatchUniformNegativeSampler(
            num_items,
            num_neg_samples,
            user_item_dict,
            device,
            index_batch_size=kwargs.get(
                "mixed_index_batch_size", kwargs.get("train_batch_size", 1024)
            ),
        )

    elif strategy == "dns":
        return DNSNegativeSampler(
            num_items,
            num_neg_samples,
            user_item_dict,
            model,
            device,
            candidate_pool_size=kwargs.get("candidate_pool_size", 100),
            temperature=kwargs.get("dns_temperature", 0.1),
        )

    elif strategy == "curriculum":
        return CurriculumNegativeSampler(
            num_items,
            num_neg_samples,
            user_item_dict,
            model,
            device,
            candidate_pool_size=kwargs.get("candidate_pool_size", 100),
            start_hard_ratio=kwargs.get("curriculum_start_ratio", 0.0),
            end_hard_ratio=kwargs.get("curriculum_end_ratio", 0.8),
            warmup_epochs=kwargs.get("curriculum_warmup_epochs", 10),
        )

    elif strategy == "debiased":
        return DebiasedNegativeSampler(
            num_items,
            num_neg_samples,
            user_item_dict,
            device,
            tau_plus=kwargs.get("tau_plus", 0.05),
        )

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
