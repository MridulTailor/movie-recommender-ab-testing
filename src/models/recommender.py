from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Train the recommender on the given DataFrame."""
        pass

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        """Return a list of top-n movie IDs to recommend."""
        pass

class PopularityRecommender(BaseRecommender):
    """
    Recommends the most popular movies (highest number of ratings) to everyone.
    Good baseline for Cold Start.
    """
    def __init__(self):
        self.popular_movies = []

    def fit(self, df: pd.DataFrame):
        # Count ratings per movie
        self.popular_movies = df.groupby('movieId').size().sort_values(ascending=False).index.tolist()

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        return self.popular_movies[:n]

class SVDRecommender(BaseRecommender):
    """
    Collaborative Filtering using Matrix Factorization (Truncated SVD).
    """
    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.pivot_table = None
        self.model = None
        self.movie_index = None
    
    def fit(self, df: pd.DataFrame):
        # Create User-Item Matrix
        # Fill NaNs with 0 (assuming unrated = 0 interest, or just centering)
        self.pivot_table = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Decompose
        # Transpose so rows are movies (for item-item similarity if needed) or just user vectors
        # SVD on User-Item matrix: U * Sigma * Vt
        # We want to approximate the interaction matrix.
        
        self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_features = self.model.fit_transform(self.pivot_table) # Users x Components
        self.item_features = self.model.components_ # Components x Items
        
        self.movie_ids = self.pivot_table.columns.tolist()
        self.user_ids = self.pivot_table.index.tolist()
        self.user_id_map = {uid: i for i, uid in enumerate(self.user_ids)}

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        if user_id not in self.user_id_map:
            # Fallback to empty or could handle with a fallback model
            return []
        
        # Reconstruct score for this user
        user_idx = self.user_id_map[user_id]
        user_vector = self.user_features[user_idx] # (n_components,)
        
        # Dot product with all items
        scores = np.dot(user_vector, self.item_features) # (n_items,)
        
        # Get top indices
        top_indices = scores.argsort()[::-1]
        
        # Filter out movies potentially already seen? 
        # For this simple verification, let's just return top scores. 
        # In prod, we'd filter out seen items.
        
        # Map back to movieId
        top_movie_ids = [self.movie_ids[i] for i in top_indices[:n]]
        return top_movie_ids

if __name__ == "__main__":
    # Test
    from src.data.loader import get_merged_data
    df = get_merged_data()
    
    print("Training Popularity Model...")
    pop = PopularityRecommender()
    pop.fit(df)
    print("Popularity Recs:", pop.recommend(1)[:5])
    
    print("Training SVD Model...")
    svd = SVDRecommender(n_components=20)
    svd.fit(df)
    print("SVD Recs for User 1:", svd.recommend(1)[:5])
