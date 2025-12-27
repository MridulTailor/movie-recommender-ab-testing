import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseRecommender

class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=20):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Rating prediction
        )
        
    def forward(self, user_indices, item_indices):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        # Concatenate user and item embeddings
        vector = torch.cat([user_embed, item_embed], dim=-1)
        output = self.fc_layers(vector)
        return output.squeeze()

class NeuralCFRecommender(BaseRecommender):
    """
    Neural Collaborative Filtering using PyTorch.
    Simple MVP: Matrix Factorization style MLP.
    """
    def __init__(self, embedding_dim: int = 20, epochs: int = 5):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.model = None
        self.user_id_map = None
        self.item_id_map = None
        self.inv_item_map = None
        self.device = torch.device("cpu") # Keep simple for now

    def fit(self, df: pd.DataFrame):
        # Mappings
        user_ids = df['userId'].unique().tolist()
        item_ids = df['movieId'].unique().tolist()
        
        self.user_id_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_id_map = {iid: i for i, iid in enumerate(item_ids)}
        self.inv_item_map = {v: k for k, v in self.item_id_map.items()}
        
        # Prepare Data
        X_users = torch.tensor([self.user_id_map[u] for u in df['userId'].values], dtype=torch.long)
        X_items = torch.tensor([self.item_id_map[i] for i in df['movieId'].values], dtype=torch.long)
        y_ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        
        dataset = TensorDataset(X_users, X_items, y_ratings)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        # Initialize Model
        num_users = len(user_ids)
        num_items = len(item_ids)
        self.model = NCFModel(num_users, num_items, self.embedding_dim).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_users, batch_items, batch_ratings in dataloader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_users, batch_items)
                loss = criterion(outputs, batch_ratings)
                loss.backward()
                optimizer.step()

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        if user_id not in self.user_id_map:
            return []
            
        internal_user_id = self.user_id_map[user_id]
        
        # Prepare all items for prediction
        all_item_indices = torch.tensor(list(self.item_id_map.values()), dtype=torch.long).to(self.device)
        user_indices = torch.tensor([internal_user_id] * len(all_item_indices), dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(user_indices, all_item_indices)
            
        # Top N
        top_indices = torch.argsort(predictions, descending=True)[:n]
        top_item_internal_indices = all_item_indices[top_indices].cpu().numpy()
        
        recs = [self.inv_item_map[idx] for idx in top_item_internal_indices]
        return recs
