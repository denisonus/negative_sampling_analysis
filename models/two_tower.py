"""Two-Tower Recommender Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """Encoder tower for user or item IDs with optional side features."""

    def __init__(
        self,
        input_dim,
        embedding_size,
        hidden_size,
        num_layers,
        dropout,
        feature_schema=None,
        feature_tensors=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_size)
        self.feature_schema = list(feature_schema or [])
        self.feature_tensors = dict(feature_tensors or {})

        if self.feature_schema and not self.feature_tensors:
            raise ValueError("Feature schema requires feature tensors")

        self.feature_names = []
        self.feature_types = {}
        self.feature_buffer_names = {}
        self.side_embeddings = nn.ModuleDict()

        for index, spec in enumerate(self.feature_schema):
            name = spec["name"]
            if name not in self.feature_tensors:
                raise ValueError(f"Missing feature tensor for '{name}'")

            tensor = self.feature_tensors[name].long()
            buffer_name = f"feature_tensor_{index}"
            self.register_buffer(buffer_name, tensor)
            self.feature_buffer_names[name] = buffer_name
            self.feature_names.append(name)
            self.feature_types[name] = spec["type"]
            self.side_embeddings[name] = nn.Embedding(
                spec["num_embeddings"], embedding_size, padding_idx=0
            )

        input_size = embedding_size * (1 + len(self.feature_names))
        layers = []
        in_size = input_size
        for _ in range(num_layers):
            layers.extend(
                [nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            in_size = hidden_size

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_size, embedding_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_feature_tensor(self, name):
        return getattr(self, self.feature_buffer_names[name])

    @staticmethod
    def _pool_token_sequence(embeddings, token_ids):
        mask = token_ids.ne(0).unsqueeze(-1)
        masked_embeddings = embeddings * mask
        denom = mask.sum(dim=-2).clamp(min=1).to(embeddings.dtype)
        return masked_embeddings.sum(dim=-2) / denom

    def _encode_side_feature(self, name, entity_ids):
        feature_tensor = self._get_feature_tensor(name)
        feature_values = feature_tensor[entity_ids]
        feature_embeddings = self.side_embeddings[name](feature_values)

        if self.feature_types[name] == "token":
            return feature_embeddings
        if self.feature_types[name] == "token_seq":
            return self._pool_token_sequence(feature_embeddings, feature_values)
        raise ValueError(f"Unsupported feature type: {self.feature_types[name]}")

    def forward(self, x):
        parts = [self.embedding(x)]
        for name in self.feature_names:
            parts.append(self._encode_side_feature(name, x))

        tower_input = torch.cat(parts, dim=-1)
        hidden = self.mlp(tower_input) if len(self.mlp) > 0 else tower_input
        return F.normalize(self.output_layer(hidden), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Two-Tower Model with optional side-feature aware encoders."""

    def __init__(
        self,
        num_users,
        num_items,
        embedding_size=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        user_feature_schema=None,
        user_feature_tensors=None,
        item_feature_schema=None,
        item_feature_tensors=None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        self.user_tower = Tower(
            num_users,
            embedding_size,
            hidden_size,
            num_layers,
            dropout,
            feature_schema=user_feature_schema,
            feature_tensors=user_feature_tensors,
        )
        self.item_tower = Tower(
            num_items,
            embedding_size,
            hidden_size,
            num_layers,
            dropout,
            feature_schema=item_feature_schema,
            feature_tensors=item_feature_tensors,
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

    def compute_loss(
        self,
        user_ids,
        pos_item_ids,
        neg_item_ids,
        neg_log_probs=None,
        tau_plus=None,
    ):
        """Compute contrastive loss (InfoNCE / Sampled Softmax).

        Args:
            user_ids: User IDs tensor
            pos_item_ids: Positive item IDs tensor
            neg_item_ids: Negative item IDs tensor (batch_size, num_neg)
            neg_log_probs: Optional log sampling probabilities for bias correction.
                           Shape: (batch_size, num_neg). When provided, applies
                           logQ correction: logits -= log(Q(neg)) to debias
                           non-uniform sampling distributions.
            tau_plus: Optional positive class prior for debiased contrastive loss.
                      Corrects for false negatives among sampled negatives.
        """
        user_emb = self.get_user_embedding(user_ids)
        pos_item_emb = self.get_item_embedding(pos_item_ids)
        neg_item_emb = self.get_item_embedding(neg_item_ids)

        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_item_emb, user_emb.unsqueeze(-1)).squeeze(-1)

        temp = self.temperature.clamp(min=0.01)
        pos_logits = pos_scores / temp
        neg_logits = neg_scores / temp

        if neg_log_probs is not None:
            neg_logits = neg_logits - neg_log_probs

        if tau_plus is not None:
            N = neg_logits.size(1)
            neg_exp = torch.exp(neg_logits)
            pos_exp = torch.exp(pos_logits)
            neg_sum = neg_exp.sum(dim=1, keepdim=True)
            debiased_neg = (neg_sum - N * tau_plus * pos_exp) / (1 - tau_plus)
            debiased_neg = debiased_neg.clamp(
                min=N
                * torch.exp(torch.tensor(-1.0 / temp.item(), device=neg_logits.device))
            )
            loss = -torch.log(pos_exp / (pos_exp + debiased_neg) + 1e-8)
            return loss.mean()

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
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

        item_emb = self.get_item_embedding(item_ids)
        return torch.sum(user_emb * item_emb, dim=-1)
