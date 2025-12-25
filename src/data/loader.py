import os
import requests
import zipfile
import io
import pandas as pd
from pathlib import Path

# Constants
MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def download_movielens_small():
    """Downloads and extracts the MovieLens Latest Small dataset."""
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    if (RAW_DIR / "ml-latest-small").exists():
        print("Dataset already downloaded.")
        return

    print(f"Downloading {MOVIELENS_SMALL_URL}...")
    response = requests.get(MOVIELENS_SMALL_URL)
    response.raise_for_status()

    print("Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(RAW_DIR)
    
    print("Download and extraction complete.")

def load_ratings():
    """Loads ratings.csv into a generic DataFrame."""
    ratings_path = RAW_DIR / "ml-latest-small" / "ratings.csv"
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings file not found at {ratings_path}. Run download_movielens_small() first.")
    
    return pd.read_csv(ratings_path)

def load_movies():
    """Loads movies.csv into a DataFrame."""
    movies_path = RAW_DIR / "ml-latest-small" / "movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies file not found at {movies_path}. Run download_movielens_small() first.")
    
    return pd.read_csv(movies_path)

def get_merged_data():
    """Returns a merged DataFrame of ratings and movies."""
    ratings = load_ratings()
    movies = load_movies()
    
    # Merge on movieId
    df = pd.merge(ratings, movies, on="movieId")
    return df

if __name__ == "__main__":
    download_movielens_small()
    df = get_merged_data()
    print(f"Loaded {len(df)} ratings.")
    print(df.head())
