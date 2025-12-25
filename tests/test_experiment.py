import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pytest
from src.experiment.ab_test import ExperimentEngine

def test_bucket_assignment_deterministic():
    """Test that the same user always gets the same bucket."""
    engine = ExperimentEngine(salt="test_salt")
    
    user_id = 12345
    bucket1 = engine.assign_bucket(user_id)
    bucket2 = engine.assign_bucket(user_id)
    
    assert bucket1 == bucket2
    assert bucket1 in ["control", "treatment"]

def test_bucket_distribution():
    """Test that buckets are roughly distributed 50/50 over many users."""
    engine = ExperimentEngine(salt="dist_test")
    
    control_count = 0
    treatment_count = 0
    n_users = 1000
    
    for i in range(n_users):
        bucket = engine.assign_bucket(i)
        if bucket == "control":
            control_count += 1
        else:
            treatment_count += 1
            
    # Check if roughly balanced (e.g., within 40-60%)
    ratio = control_count / n_users
    assert 0.4 < ratio < 0.6
