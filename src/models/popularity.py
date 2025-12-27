import pandas as pd
from .base import BaseRecommender

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
