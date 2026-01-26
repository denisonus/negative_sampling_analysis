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
            layers.extend([nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)])
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
    
    def __init__(self, num_users, num_items, embedding_size=64, 
                 hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        self.user_tower = Tower(num_users, embedding_size, hidden_size, num_layers, dropout)
        self.item_tower = Tower(num_items, embedding_size, hidden_size, num_layers, dropout)
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
    
    def compute_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """Compute contrastive loss (InfoNCE / Sampled Softmax)."""
        user_emb = self.get_user_embedding(user_ids)
        pos_item_emb = self.get_item_embedding(pos_item_ids)
        neg_item_emb = self.get_item_embedding(neg_item_ids)
        
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_item_emb, user_emb.unsqueeze(-1)).squeeze(-1)
        
        logits = torch.cat([pos_scores, neg_scores], dim=-1) / self.temperature.clamp(min=0.01)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)
    
    def compute_bpr_loss(self, user_ids, pos_item_ids, neg_item_ids):
        """Compute BPR (Bayesian Personalized Ranking) loss."""
        user_emb = self.get_user_embedding(user_ids)
        pos_item_emb = self.get_item_embedding(pos_item_ids)
        neg_item_emb = self.get_item_embedding(neg_item_ids)
        
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1, keepdim=True)
        neg_scores = torch.bmm(neg_item_emb, user_emb.unsqueeze(-1)).squeeze(-1)
        
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    
    def predict(self, user_ids, item_ids=None):
        """Predict scores for user-item pairs or all items."""
        user_emb = self.get_user_embedding(user_ids)
        
        if item_ids is None:
            all_items = torch.arange(self.num_items, device=user_ids.device)
            all_item_emb = self.get_item_embedding(all_items)
            return torch.matmul(user_emb, all_item_emb.t())
        else:
            item_emb = self.get_item_embedding(item_ids)
            return torch.sum(user_emb * item_emb, dim=-1)
