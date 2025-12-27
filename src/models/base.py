from abc import ABC, abstractmethod
import pandas as pd

class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """Train the recommender on the given DataFrame."""
        pass

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[int]:
        """Return a list of top-n movie IDs to recommend."""
        pass
