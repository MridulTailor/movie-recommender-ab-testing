import pytest
import pandas as pd
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.recommender import PopularityRecommender, SVDRecommender

@pytest.fixture
def mock_data():
    """Creates a small mock dataset for testing."""
    data = {
        'userId': [1, 1, 1, 2, 2, 3, 3, 4],
        'movieId': [101, 102, 103, 101, 104, 102, 105, 101],
        'rating': [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 1.0, 5.0]
    }
    return pd.DataFrame(data)

def test_popularity_recommender(mock_data):
    """Test that PopularityRecommender returns the most frequent items."""
    model = PopularityRecommender()
    model.fit(mock_data)
    
    # Movie 101 appears 3 times (most popular)
    # Movie 102 appears 2 times
    # Others appear 1 time
    
    recs = model.recommend(user_id=999, n=2) # User ID doesn't matter for popularity
    assert recs == [101, 102]
    
    recs_all = model.recommend(user_id=999, n=5)
    assert recs_all[0] == 101

def test_svd_recommender_fit_and_recommend(mock_data):
    """Test SVD training and basic recommendation structure."""
    model = SVDRecommender(n_components=2) # Small components for small data
    model.fit(mock_data)
    
    # Check if things were created
    assert model.pivot_table is not None
    assert model.model is not None
    
    # Test Existing User
    recs = model.recommend(user_id=1, n=2)
    assert len(recs) == 2
    assert isinstance(recs, list)
    assert all(isinstance(x, (int, np.integer)) for x in recs)

def test_svd_recommender_cold_start(mock_data):
    """Test behavior for a user not in the training set."""
    model = SVDRecommender(n_components=2)
    model.fit(mock_data)
    
    # User 999 does not exist
    recs = model.recommend(user_id=999)
    assert recs == [] # Should be empty list as per current implementation
