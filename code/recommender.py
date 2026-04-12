"""
CSCI 335/635
Movie Recommendation Project

File: recommender.py

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# Load Dataset
# ============================================================

RATINGS_FILE = "..data/ml-100k/u.data"
MOVIES_FILE = "..data/ml-100k/u.item"

ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]

movie_cols = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

ratings = None
movies = None

# ============================================================
# Dataset Check
# ============================================================



# ============================================================
# Merge and Preprocess Data
# ============================================================

# ============================================================
# Train Test Split
# ============================================================



# ============================================================
# Baseline Model Preparation
# ============================================================


# ============================================================
# Baseline Model
# ============================================================


# ============================================================
# Traditional Models
# ============================================================


# ============================================================
# Neural Network
# ============================================================



# ============================================================
# Recommendation Output
# ============================================================



# ============================================================
# Model Comparison
# ============================================================

def main():

    print()

    # 1. load ratings and movies
    # 2. preprocess data
    # 3. split into training and testing sets
    # 4. complete baseline model first
    # 5. evaluate baseline model
    # 6. add traditional models
    # 7. add neural model
    # 8. compare models



if __name__ == "__main__":
    main()