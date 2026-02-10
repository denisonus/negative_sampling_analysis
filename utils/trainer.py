"""Training Loop for Two-Tower Model."""

import torch
import torch.optim as optim
import time
from typing import Any, Union
from tqdm import tqdm

from samplers.base import SamplingResult


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
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
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

            loss = self.model.compute_loss(
                user_ids, pos_item_ids, neg_item_ids, neg_log_probs
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
        patience = self.config.get("patience", 10)

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

                valid_metric_name = self.config.get("valid_metric", "ndcg@10").lower()
                valid_metric = metrics.get(valid_metric_name, 0)
                self.valid_metrics.append(valid_metric)

                print(f"  Validation: {metrics}")

                # Learning rate scheduling
                self.scheduler.step(valid_metric)

                # Early stopping
                if valid_metric > best_metric:
                    best_metric = valid_metric
                    best_epoch = epoch
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
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
    """Trainer with in-batch negative sampling."""

    def train_epoch(self, train_loader, epoch=0):
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_sampling_time = 0  # No separate sampling for in-batch
        epoch_training_time = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for user_ids, pos_item_ids in pbar:
            user_ids = user_ids.to(self.device)
            pos_item_ids = pos_item_ids.to(self.device)
            batch_size = user_ids.size(0)

            train_start = time.time()
            user_emb = self.model.get_user_embedding(user_ids)
            item_emb = self.model.get_item_embedding(pos_item_ids)

            logits = torch.matmul(user_emb, item_emb.t())
            logits = logits / self.model.temperature.clamp(min=0.01)

            labels = torch.arange(batch_size, device=self.device)
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
