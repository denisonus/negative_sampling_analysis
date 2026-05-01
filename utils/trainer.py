"""Training Loop for Two-Tower Model."""

import copy
import torch
import torch.optim as optim
import time
from typing import Any, Union
from tqdm import tqdm

from samplers.base import SamplingResult
from samplers.debiased import DebiasedNegativeSampler
from samplers.mixed_in_batch_uniform import MixedInBatchUniformNegativeSampler


def _select_validation_log_metrics(metrics, valid_metric_name):
    """Pick a compact, non-empty validation log for the configured metric."""
    if not metrics:
        return {}

    valid_metric_name = valid_metric_name.lower()
    configured_k = valid_metric_name.rsplit("@", 1)[-1] if "@" in valid_metric_name else None
    selected = {}

    if configured_k is not None:
        selected = {
            key: value for key, value in metrics.items() if key.endswith(f"@{configured_k}")
        }

    if not selected and valid_metric_name in metrics:
        selected = {valid_metric_name: metrics[valid_metric_name]}

    return selected or metrics


def _build_interaction_matrix(user_item_dict, num_items, device):
    """Build a sparse user-item matrix for fast false-negative masking."""
    rows, cols = [], []
    for uid, items in user_item_dict.items():
        for iid in items:
            rows.append(uid)
            cols.append(iid)

    n_users = max(user_item_dict.keys()) + 1 if user_item_dict else 0
    return (
        torch.sparse_coo_tensor(
            torch.tensor([rows, cols], dtype=torch.long),
            torch.ones(len(rows), dtype=torch.float32),
            size=(n_users, num_items),
        )
        .coalesce()
        .to(device)
    )


class Trainer:
    """Trainer for Two-Tower model with customizable negative sampling."""

    def __init__(
        self,
        model: Any,
        sampler: Any,
        config: dict,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model = model.to(device)
        self.sampler = sampler
        self.config = config
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        self.train_losses = []
        self.valid_metrics = []

        # Timing metrics
        self.epoch_times = []
        self.sampling_times = []
        self.training_times = []

    def train_epoch(self, train_loader, epoch=0):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_sampling_time = 0
        epoch_training_time = 0

        self.sampler.set_epoch(epoch)
        self.sampler.set_model(self.model)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for user_ids, pos_item_ids in pbar:
            user_ids = user_ids.to(self.device)
            pos_item_ids = pos_item_ids.to(self.device)

            # Time negative sampling
            sample_start = time.time()
            sample_result = self.sampler.sample(user_ids, pos_item_ids)

            # Handle both plain tensor and SamplingResult
            if isinstance(sample_result, SamplingResult):
                neg_item_ids = sample_result.neg_items.to(self.device)
                neg_log_probs = sample_result.log_probs
                if neg_log_probs is not None:
                    neg_log_probs = neg_log_probs.to(self.device)
            else:
                neg_item_ids = sample_result.to(self.device)
                neg_log_probs = None

            epoch_sampling_time += time.time() - sample_start

            # Time training step
            train_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            # Pass extra kwargs for specialized samplers
            extra_kwargs = {}
            if isinstance(self.sampler, DebiasedNegativeSampler):
                extra_kwargs["tau_plus"] = self.sampler.last_tau_plus

            loss = self.model.compute_loss(
                user_ids, pos_item_ids, neg_item_ids, neg_log_probs, **extra_kwargs
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_training_time += time.time() - train_start

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.sampling_times.append(epoch_sampling_time)
        self.training_times.append(epoch_training_time)

        return avg_loss

    def fit(self, train_loader, valid_loader=None, evaluator=None, epochs=50):
        """Full training loop with optional validation and early stopping."""
        best_metric = 0
        best_epoch = 0
        patience_counter = 0
        patience = self.config["patience"]

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time
            self.epoch_times.append(epoch_time)

            print(
                f"Epoch {epoch}: Loss = {train_loss:.4f}, Time = {epoch_time:.2f}s "
                f"(Sampling: {self.sampling_times[-1]:.2f}s, Training: {self.training_times[-1]:.2f}s)"
            )

            # Validation
            if valid_loader is not None and evaluator is not None:
                metrics = evaluator.evaluate(self.model, valid_loader)

                valid_metric_name = self.config["valid_metric"].lower()
                valid_metric = metrics.get(valid_metric_name, 0)
                self.valid_metrics.append({k: float(v) for k, v in metrics.items()})

                logged_metrics = {
                    key: f"{value:.4f}"
                    for key, value in _select_validation_log_metrics(
                        metrics, valid_metric_name
                    ).items()
                }
                print(f"Validation: {logged_metrics}")

                # Learning rate scheduling
                self.scheduler.step(valid_metric)

                # Early stopping
                if valid_metric > best_metric:
                    best_metric = valid_metric
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                    break

        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)

        return {
            "train_losses": self.train_losses,
            "valid_metrics": self.valid_metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epoch_times": self.epoch_times,
            "sampling_times": self.sampling_times,
            "training_times": self.training_times,
            "total_time": sum(self.epoch_times),
            "total_sampling_time": sum(self.sampling_times),
            "total_training_time": sum(self.training_times),
        }


class InBatchTrainer(Trainer):
    """Trainer with in-batch negative sampling.
    
    Optionally applies logQ correction
    """

    def __init__(self, model, sampler, config, device, item_popularity=None):
        super().__init__(model, sampler, config, device)
        self.logq_correction = config["logq_correction"]
        self._log_q = None
        if self.logq_correction and item_popularity is not None:
            import numpy as np

            freq = np.array(item_popularity, dtype=np.float64)
            prob = freq / freq.sum()
            self._log_q = torch.from_numpy(np.log(prob + 1e-10)).float().to(device)

        self._interaction = _build_interaction_matrix(
            sampler.user_item_dict, sampler.num_items, device
        )

    def train_epoch(self, train_loader, epoch=0):
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_sampling_time = 0
        epoch_training_time = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for user_ids, pos_item_ids in pbar:
            user_ids = user_ids.to(self.device)
            pos_item_ids = pos_item_ids.to(self.device)
            batch_size = user_ids.size(0)

            sample_start = time.time()
            user_emb = self.model.get_user_embedding(user_ids)
            item_emb = self.model.get_item_embedding(pos_item_ids)

            logits = torch.matmul(user_emb, item_emb.t())
            logits = logits / self.model.temperature.clamp(min=0.01)

            # logQ correction: subtract log(Q(item_j)) from each column j
            if self.logq_correction and self._log_q is not None:
                logits = logits - self._log_q[pos_item_ids].unsqueeze(0)

            # Vectorized false-negative masking via sparse interaction matrix:
            # mask[i, j] = True when user_i has interacted with item at position j
            mask = (
                self._interaction.index_select(0, user_ids)
                .to_dense()[:, pos_item_ids]
                .bool()
            )
            # Keep diagonal (true positive for each user)
            mask.fill_diagonal_(False)
            logits[mask] = float("-inf")
            epoch_sampling_time += time.time() - sample_start

            labels = torch.arange(batch_size, device=self.device)

            train_start = time.time()
            loss = torch.nn.functional.cross_entropy(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_training_time += time.time() - train_start

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.sampling_times.append(epoch_sampling_time)
        self.training_times.append(epoch_training_time)

        return avg_loss


class MixedInBatchTrainer(InBatchTrainer):
    """Trainer for the in-batch + uniform mixed-negative variant."""

    def __init__(self, model, sampler, config, device, item_popularity=None):
        super().__init__(model, sampler, config, device, item_popularity=item_popularity)
        if not isinstance(sampler, MixedInBatchUniformNegativeSampler):
            raise TypeError("MixedInBatchTrainer requires MixedInBatchUniformNegativeSampler")

    def train_epoch(self, train_loader, epoch=0):
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_sampling_time = 0
        epoch_training_time = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for user_ids, pos_item_ids in pbar:
            user_ids = user_ids.to(self.device)
            pos_item_ids = pos_item_ids.to(self.device)
            batch_size = user_ids.size(0)

            sample_start = time.time()
            user_emb = self.model.get_user_embedding(user_ids)
            batch_item_emb = self.model.get_item_embedding(pos_item_ids)
            batch_logits = torch.matmul(user_emb, batch_item_emb.t())
            batch_logits = batch_logits / self.model.temperature.clamp(min=0.01)

            if self.logq_correction and self._log_q is not None:
                batch_logits = batch_logits - self._log_q[pos_item_ids].unsqueeze(0)

            batch_mask = (
                self._interaction.index_select(0, user_ids)
                .to_dense()[:, pos_item_ids]
                .bool()
            )
            batch_mask.fill_diagonal_(False)
            batch_logits[batch_mask] = float("-inf")

            shared_neg_ids = self.sampler.sample_shared_uniform_items(
                exclude_item_ids=pos_item_ids
            )
            if shared_neg_ids.numel() > 0:
                shared_item_emb = self.model.get_item_embedding(shared_neg_ids)
                shared_logits = torch.matmul(user_emb, shared_item_emb.t())
                shared_logits = shared_logits / self.model.temperature.clamp(min=0.01)

                shared_mask = (
                    self._interaction.index_select(0, user_ids)
                    .to_dense()[:, shared_neg_ids]
                    .bool()
                )
                shared_logits[shared_mask] = float("-inf")
                logits = torch.cat([batch_logits, shared_logits], dim=1)
            else:
                logits = batch_logits

            epoch_sampling_time += time.time() - sample_start

            labels = torch.arange(batch_size, device=self.device)

            train_start = time.time()
            loss = torch.nn.functional.cross_entropy(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_training_time += time.time() - train_start

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.sampling_times.append(epoch_sampling_time)
        self.training_times.append(epoch_training_time)

        return avg_loss
