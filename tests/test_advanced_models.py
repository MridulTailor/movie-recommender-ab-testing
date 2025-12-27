import pytest
import pandas as pd
import numpy as np
from src.models import SurpriseRecommender, NeuralCFRecommender

@pytest.fixture
def mock_data():
    """Creates a small mock dataset for testing."""
    data = {
        'userId': [1, 1, 1, 2, 2, 3, 3, 4],
        'movieId': [101, 102, 103, 101, 104, 102, 105, 101],
        'rating': [5.0, 3.0, 4.0, 2.0, 5.0, 4.0, 1.0, 5.0]
    }
    return pd.DataFrame(data)

def test_surprise_recommender(mock_data):
    """Test Surprise model training and recommendation."""
    # Using small params for speed
    model = SurpriseRecommender(n_factors=5, n_epochs=5)
    model.fit(mock_data)
    
    # Recommend for user 1
    recs = model.recommend(user_id=1, n=2)
    assert len(recs) <= 2 # Might be less if not enough items
    assert isinstance(recs, list)
    
    # Cold start
    recs_empty = model.recommend(user_id=999)
    assert recs_empty == []

def test_neuralcf_recommender(mock_data):
    """Test NeuralCF model training and recommendation."""
    # Run with very few epochs for speed
    model = NeuralCFRecommender(embedding_dim=4, epochs=1)
    model.fit(mock_data)
    
    # Recommend for user 1
    recs = model.recommend(user_id=1, n=2)
    assert len(recs) == 2
    assert isinstance(recs, list)
    
    # Cold start
    recs_empty = model.recommend(user_id=999)
    assert recs_empty == []
