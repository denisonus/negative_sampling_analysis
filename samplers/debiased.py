"""Debiased Contrastive negative sampling.

Corrects for the fact that uniformly sampled "negatives" may actually be
unlabeled positives, using importance weighting to debias the contrastive loss.

Reference: Chuang et al., "Debiased Contrastive Learning" (NeurIPS 2020)
"""

import torch
import numpy as np
from typing import Set, Dict

from .base import NegativeSampler, Device


class DebiasedNegativeSampler(NegativeSampler):
    """Debiased Contrastive negative sampling.

    Samples negatives uniformly and estimates the probability that a sampled
    negative is a true negative (not an unlabeled positive). This produces
    a debiased score that the loss function can use.

    The key idea: with N total items and tau as the estimated positive class
    prior, the debiased negative score is:
        g = max(score_neg - tau * score_pos, e^(-1/t))
    where the clipping prevents degenerate solutions.
    """

    def __init__(
        self,
        num_items: int,
        num_neg_samples: int,
        user_item_dict: Dict[int, Set[int]],
        device: Device = "cpu",
        tau_plus: float = 0.05,
    ):
        super().__init__(num_items, num_neg_samples, user_item_dict, device)
        self.name = "debiased"
        self.tau_plus = tau_plus  # Estimated positive class prior

    def sample(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Sample negatives uniformly. Debiasing is applied at the loss level.

        The debiasing correction weights are stored in self.last_tau_plus
        for use by the trainer.
        """
        batch_size = user_ids.size(0)

        oversample = max(self.num_neg_samples * 3, self.num_neg_samples + 50)
        candidates = np.random.randint(0, self.num_items, size=(batch_size, oversample))

        neg_items = np.zeros((batch_size, self.num_neg_samples), dtype=np.int64)
        user_ids_np = user_ids.cpu().numpy()

        for i in range(batch_size):
            positives = self._get_positives(user_ids_np[i])
            row = candidates[i]
            mask = np.isin(row, list(positives), invert=True)
            valid = row[mask]

            if len(valid) >= self.num_neg_samples:
                neg_items[i] = valid[: self.num_neg_samples]
            else:
                neg_items[i, : len(valid)] = valid
                idx = len(valid)
                while idx < self.num_neg_samples:
                    c = np.random.randint(0, self.num_items)
                    if c not in positives:
                        neg_items[i, idx] = c
                        idx += 1

        # Store tau_plus for debiased loss computation
        self.last_tau_plus = self.tau_plus

        return torch.from_numpy(neg_items).to(self.device)
