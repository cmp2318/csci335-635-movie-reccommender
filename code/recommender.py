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

RATINGS_FILE = "data/ml-100k/u.data"
MOVIES_FILE = "data/ml-100k/u.item"

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

"""
performs a quick check and printout of stats from the 
dataset, including number of unique users, movies, and missing values
"""

def dataset_check(ratings_df, movies_df):

    print("=== Dataset Check ===")
    print()

    print("Ratings shape:", ratings_df.shape)
    print("Movies shape:", movies_df.shape)
    print()

    print("Ratings preview:")
    print(ratings_df.head())
    print()

    print("Movies preview:")
    print(movies_df[["movie_id", "title"]].head())
    print()

    print("Unique users:", ratings_df["user_id"].nunique())
    print("Unique movies rated:", ratings_df["movie_id"].nunique())
    print("Total ratings:", len(ratings_df))
    print()

    print("Missing values in ratings:")
    print(ratings_df.isnull().sum())
    print()

    print("Missing values in movies:")
    print(movies_df.isnull().sum())
    print()

# ============================================================
# Merge and Preprocess Data
# ============================================================

"""
merges the ratings data with the movie titles using
movie_id, then keeps only the columns needed for the
current stage
"""

def merge_and_preprocess(ratings_df, movies_df):

    # merge ratings with movie titles
    movie_ratings = ratings_df.merge(
        movies_df[["movie_id", "title"]],
        on="movie_id",
        how="left"
    )

    # only relevant columns right now
    movie_ratings = movie_ratings[
        ["user_id", "movie_id", "title", "rating", "timestamp"]
    ]

    return movie_ratings

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

    global ratings
    global movies


    # process the u.item and u.data file with pandas

    ratings = pd.read_csv("../data/ml-100k/u.data",
                      sep="\t",
                      names=["user_id", "movie_id", "rating", "timestamp"])

    movies = pd.read_csv("../data/ml-100k/u.item",
                         sep="|",
                         names=movie_cols,
                         encoding="latin-1")

    print("Datasets loaded")
    print()


    # dataset check step

    dataset_check(ratings, movies)

    # merge and preprocess step

    movie_ratings = merge_and_preprocess(ratings, movies)

    print("=== Merge and Preprocess Data ===")
    print()

    print("Merged shape:", movie_ratings.shape)
    print()

    print("Merged preview:")
    print(movie_ratings.head())
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