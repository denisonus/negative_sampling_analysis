"""Two-Tower Recommender Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """Encoder tower for user or item features."""

    def __init__(self, input_dim, embedding_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_size)

        layers = []
        in_size = embedding_size
        for _ in range(num_layers):
            layers.extend(
                [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            in_size = hidden_size

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, embedding_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x)
        hidden = self.mlp(emb)
        return F.normalize(self.output_layer(hidden), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Two-Tower Model with user and item encoders."""

    def __init__(
        self,
        num_users,
        num_items,
        embedding_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        self.user_tower = Tower(
            num_users, embedding_size, hidden_size, num_layers, dropout
        )
        self.item_tower = Tower(
            num_items, embedding_size, hidden_size, num_layers, dropout
        )
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def get_user_embedding(self, user_ids):
        return self.user_tower(user_ids)

    def get_item_embedding(self, item_ids):
        return self.item_tower(item_ids)

    def forward(self, user_ids, item_ids):
        user_emb = self.get_user_embedding(user_ids)
        item_emb = self.get_item_embedding(item_ids)

        if item_emb.dim() == 3:
            user_emb = user_emb.unsqueeze(1)
        scores = torch.sum(user_emb * item_emb, dim=-1)

        return scores / self.temperature.clamp(min=0.01)

    def compute_loss(self, user_ids, pos_item_ids, neg_item_ids, neg_log_probs=None):
        """Compute contrastive loss (InfoNCE / Sampled Softmax).

        Args:
            user_ids: User IDs tensor
            pos_item_ids: Positive item IDs tensor
            neg_item_ids: Negative item IDs tensor (batch_size, num_neg)
            neg_log_probs: Optional log sampling probabilities for bias correction.
                           Shape: (batch_size, num_neg). When provided, applies
                           logQ correction: logits -= log(Q(neg)) to debias
                           non-uniform sampling distributions.
        """
        user_emb = self.get_user_embedding(user_ids)
        pos_item_emb = self.get_item_embedding(pos_item_ids)
        neg_item_emb = self.get_item_embedding(neg_item_ids)

        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_item_emb, user_emb.unsqueeze(-1)).squeeze(-1)

        # Temperature-scaled logits
        logits = torch.cat([pos_scores, neg_scores], dim=-1) / self.temperature.clamp(
            min=0.01
        )

        # Apply logQ correction
        if neg_log_probs is not None:
            # Subtract log(Q) from negative logits only (index 1 onwards)
            logits[:, 1:] = logits[:, 1:] - neg_log_probs

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def get_all_item_embeddings(self):
        """Compute all item embeddings. Call once before evaluation."""
        all_items = torch.arange(self.num_items, device=next(self.parameters()).device)
        return self.get_item_embedding(all_items)

    def predict(self, user_ids, item_ids=None, all_item_emb=None):
        """Predict scores for user-item pairs or all items.

        Args:
            user_ids: User IDs to predict for
            item_ids: Optional specific item IDs (if None, scores all items)
            all_item_emb: Pre-computed item embeddings (optional, for efficiency)
        """
        user_emb = self.get_user_embedding(user_ids)

        if item_ids is None:
            if all_item_emb is None:
                all_item_emb = self.get_all_item_embeddings()
            return torch.matmul(user_emb, all_item_emb.t())
        else:
            item_emb = self.get_item_embedding(item_ids)
            return torch.sum(user_emb * item_emb, dim=-1)
