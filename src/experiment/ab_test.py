import hashlib
import numpy as np
import pandas as pd
from typing import Literal

from src.models.recommender import BaseRecommender, PopularityRecommender, SVDRecommender

class ExperimentEngine:
    def __init__(self, salt: str = "movie_experiment_v1"):
        self.salt = salt

    def assign_bucket(self, user_id: int) -> Literal["control", "treatment"]:
        """
        Deterministically assigns a user to 'control' or 'treatment' based on user_id.
        Uses MD5 hashing for stability.
        """
        # Create a hash of the user_id + salt
        hash_input = f"{user_id}{self.salt}".encode('utf-8')
        hash_val = hashlib.md5(hash_input).hexdigest()
        
        # Convert first few chars to int to determine modulo
        # 50/50 split
        if int(hash_val, 16) % 2 == 0:
            return "control"
        else:
            return "treatment"

class Simulator:
    def __init__(self, df_ground_truth: pd.DataFrame):
        """
        df_ground_truth: The dataframe containing REAL user ratings (test set).
        """
        self.ground_truth = df_ground_truth.set_index(['userId', 'movieId'])
        self.logs = []

    def simulate_user_visit(self, user_id: int, recommender: BaseRecommender, group: str):
        """
        Simulates one user visiting the site.
        1. Get Recs
        2. Check against Ground Truth (did they actually rate any of these movies?)
        3. Log 'Exposure' and 'Conversion' (Conversion = rated >= 4.0 in future)
        """
        recs = recommender.recommend(user_id, n=5)
        
        conversion = 0
        
        # Simple Logic: A conversion happens if the user *actually* rated one of the recommended movies >= 4.0
        # In a real scenario, we'd model click probability. Here we use "Oracle" knowledge (Held-out data).
        
        for movie_id in recs:
            if (user_id, movie_id) in self.ground_truth.index:
                actual_rating = self.ground_truth.loc[(user_id, movie_id)]['rating']
                # Handle duplicate entries if any, though pivot usually handles unique
                if isinstance(actual_rating, pd.Series):
                    actual_rating = actual_rating.iloc[0]
                
                if actual_rating >= 4.0:
                    conversion = 1
                    break # Optimistic: clicked/watched at least one
        
        self.logs.append({
            "user_id": user_id,
            "group": group,
            "converted": conversion
        })

    def run_simulation(self, user_ids: list[int], control_model: BaseRecommender, treatment_model: BaseRecommender):
        engine = ExperimentEngine()
        
        print(f"Simulating visits for {len(user_ids)} users...")
        for uid in user_ids:
            group = engine.assign_bucket(uid)
            
            if group == "control":
                self.simulate_user_visit(uid, control_model, group)
            else:
                self.simulate_user_visit(uid, treatment_model, group)
                
        return pd.DataFrame(self.logs)

if __name__ == "__main__":
    from src.data.loader import get_merged_data
    
    # 1. Load Data
    full_df = get_merged_data()
    
    # 2. Split Data (Mocking Past training data vs Future ground truth)
    # Let's simple split by time or random. Random for now.
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)
    
    # 3. Train Models on TRAIN set
    print("Training Control (Popularity)...")
    control_model = PopularityRecommender()
    control_model.fit(train_df)
    
    print("Training Treatment (SVD)...")
    treatment_model = SVDRecommender(n_components=20)
    treatment_model.fit(train_df)
    
    # 4. Simulate on TEST set users (using Test set as 'Future Truth')
    test_users = test_df['userId'].unique()
    sim = Simulator(test_df)
    
    results = sim.run_simulation(test_users, control_model, treatment_model)
    
    # 5. Quick Analysis
    print("\n--- A/B Test Results ---")
    summary = results.groupby("group")['converted'].agg(['count', 'mean'])
    summary.columns = ['visitors', 'conversion_rate']
    print(summary)
