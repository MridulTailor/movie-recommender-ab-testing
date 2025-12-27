import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from .base import BaseRecommender

class SurpriseRecommender(BaseRecommender):
    """
    Collaborative Filtering using Scikit-Surprise.
    Uses Surprise's SVD implementation (Netflix Prize winner style).
    """
    def __init__(self, n_factors: int = 20, n_epochs: int = 20):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.model = None
        self.trainset = None

    def fit(self, df: pd.DataFrame):
        # Surprise requires data in a specific format
        # Reader target_scale isn't strictly enforced for prediction but good practice
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
        
        # Build full trainset
        self.trainset = data.build_full_trainset()
        
        # Train SVD
        self.model = SVD(n_factors=self.n_factors, n_epochs=self.n_epochs, random_state=42)
        self.model.fit(self.trainset)

    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        # Surprise deals with internal/raw ids automatically if we stick to 'predict'
        # But for 'recommendations', we need to predict valid items.
        
        # Check if user is known
        try:
            inner_uid = self.trainset.to_inner_uid(user_id)
        except ValueError:
            # Cold start user
            return []
            
        # Challenge: Surprise is a Rating Predictor, not inherently a Top-N Recommender
        # We need to predict rating for ALL items (or a subset) and sort.
        # Ideally, we filter out items the user has already rated.
        
        # Get all item IDs
        all_item_inner_ids = self.trainset.all_items()
        
        # Determine items user has already seen
        user_seen_items = set([j for (j, _) in self.trainset.ur[inner_uid]])
        
        # List to keep predictions
        candidates = []
        
        # Score items
        for iid in all_item_inner_ids:
            if iid not in user_seen_items:
                # Predict
                # SVD.estimate returns the estimated rating
                est_rating = self.model.estimate(inner_uid, iid)
                candidates.append((iid, est_rating))
        
        # Sort desc
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Top N
        top_candidates = candidates[:n]
        
        # Map back to raw IDs
        recs = [self.trainset.to_raw_iid(iid) for (iid, _) in top_candidates]
        
        return recs
