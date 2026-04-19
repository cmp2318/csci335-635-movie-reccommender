"""
CSCI 335/635
Movie Recommendation Project

File: neural_model.py

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Model


# ============================================================
# Load Dataset
# ============================================================

ratings = None
movies = None
users = None

ratings_cols = ["user_id", "movie_id", "rating", "timestamp"]

movie_cols = [
    "movie_id", "title", "release_date", "video_release_date", "imdb_url",
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

user_cols = ["user_id", "age", "gender", "occupation", "zip_code"]


# ============================================================
# Merge and Preprocess Neural Data
# ============================================================

"""
merges the ratings data with the movie and user
data needed for the neural network model
"""

def preprocess_neural_data(ratings_df, movies_df, users_df):

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    neural_df = None

    return neural_df


# ============================================================
# Train Test Split
# ============================================================

"""
splits the merged neural model data into training
and testing sets
"""

def split_data(neural_df):

    train_data, test_data = train_test_split(
        neural_df,
        test_size=0.20,
        random_state=35
    )

    return train_data, test_data


# ============================================================
# Neural Model Data Preparation
# ============================================================


# ============================================================
# Neural Network Model
# ============================================================

# ============================================================
# Neural Network Training
# ============================================================


# ============================================================
# Neural Network Evaluation
# ============================================================


# ============================================================
# Recommendation Output
# ============================================================


# ============================================================
# Model Comparison
# ============================================================



def main():

    global ratings
    global movies
    global users

    ratings = pd.read_csv(
        "../data/ml-100k/u.data",
        sep="\t",
        names=ratings_cols
    )

    movies = pd.read_csv(
        "../data/ml-100k/u.item",
        sep="|",
        names=movie_cols,
        encoding="latin-1"
    )

    users = pd.read_csv(
        "../data/ml-100k/u.user",
        sep="|",
        names=user_cols
    )

    print("Datasets loaded")
    print()

if __name__ == "__main__":
    main()