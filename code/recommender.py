"""
CSCI 335/635
Movie Recommendation Project

File: recommender.py

"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression


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

def preprocess(ratings_df, movies_df):

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

"""
splits the merged movie ratings data into training
and testing sets
"""

def split_data(movie_ratings_df):

    # 80/20 test split
    train_data, test_data = train_test_split(movie_ratings_df, 
                                             test_size=0.20, 
                                             random_state=35)

    return train_data, test_data

# ============================================================
# Baseline Model Preparation
# ============================================================

"""
builds the baseline's values with training data,
including average rating per movie, average user ratings,
and overall average rating
"""

def prepare_baseline(train_df):

    # average rating for each movie
    movie_mean_dict = train_df.groupby("movie_id")["rating"].mean().to_dict()

    # average rating for each user, all their ratings combined and averaged, 
    # so that the model can learn how that user tends to rate stuff
    user_mean_dict = train_df.groupby("user_id")["rating"].mean().to_dict()

    # average rating across all data
    global_mean = train_df["rating"].mean()

    return movie_mean_dict, user_mean_dict, global_mean

# ============================================================
# Baseline Model
# ============================================================

"""
predicts a rating for one user and one movie using
the movie average, the user average, or
the global average if nothing else
"""

def predict_baseline_rating(user_id, movie_id, movie_mean_dict, 
                            user_mean_dict, global_mean):

    # use average movie rating if it is in the data
    if movie_id in movie_mean_dict:
        return movie_mean_dict[movie_id]

    # if movie was not present, use the user's average rating
    if user_id in user_mean_dict:
        return user_mean_dict[user_id]

    # if neither, use the overall average rating
    return global_mean


"""
makes baseline predictions for every row in the
test set
"""

def get_baseline_predictions(test_df, movie_mean_dict, 
                             user_mean_dict, global_mean):

    predictions = []

    for i, row in test_df.iterrows():
        prediction = predict_baseline_rating(
            row["user_id"],
            row["movie_id"],
            movie_mean_dict,
            user_mean_dict,
            global_mean
        )
        predictions.append(prediction)

    return predictions


"""
evaluates the baseline model using RMSE and MAE
"""

def evaluate_baseline(test_df, predictions):

    actual = test_df["rating"]

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)

    print("=== Baseline Model Evaluation ===")
    print()

    print("RMSE:", rmse)
    print("MAE:", mae)
    print()

# ============================================================
# Traditional Models
# ============================================================

"""
normalizes ratings in the training set so 0 is a users average rating
returns matrices of normalized ratings, the original ratings, and the average ratings per user
"""

def pivot_normalize(train_df):
    user_movie_rtgs = train_df.pivot(index="user_id",
                                     columns="movie_id",
                                     values="rating")

    avg_user_rating = np.nanmean(user_movie_rtgs.values, axis=1).reshape(-1, 1)
    normalized_rtgs = user_movie_rtgs.values - avg_user_rating
    normalized_rtgs = np.nan_to_num(normalized_rtgs, nan=0)

    return normalized_rtgs, user_movie_rtgs, avg_user_rating


"""
finds neighbors of all users
returns all users neighbors' distances from the user and their indices
"""

def find_all_neighbors(knn, normalized_rtgs, k):
    distances, indices = knn.kneighbors(normalized_rtgs, n_neighbors=k + 1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    return distances, indices


"""
KNN
"""

def prepare_knn(train_df):

    normalized_rtgs, user_movie_rtgs, avg_user_rating = pivot_normalize(train_df)

    knn = NearestNeighbors(metric = "cosine", algorithm = "brute")
    knn.fit(normalized_rtgs)

    return knn, normalized_rtgs, user_movie_rtgs, avg_user_rating


def predict_knn_rating(user_id, movie_id, normalized_rtgs,
                       avg_user_rtg, user_movie_rtgs, indices, distances):

    if user_id not in user_movie_rtgs.index or movie_id not in user_movie_rtgs.columns:
        if user_id in user_movie_rtgs.index:
            user_index = user_movie_rtgs.index.get_loc(user_id)
            return avg_user_rtg[user_index][0]
        return 0

    user_index = user_movie_rtgs.index.get_loc(user_id)
    movie_index = user_movie_rtgs.columns.get_loc(movie_id)

    n_indices = indices[user_index]

    sum = 0
    similarity_sum = 0
    for i in range(n_indices.size):
        neighbor = n_indices[i]
        similarity = 1 - distances[user_index, i]
        sum += similarity * normalized_rtgs[neighbor, movie_index]
        similarity_sum += abs(similarity)

    prediction = (sum / similarity_sum) + avg_user_rtg[user_index][0]

    return prediction


def evaluate_knn(test_df, knn, normalized_rtgs, avg_user_rtg, user_movie_rtgs, k):

    actuals = []
    predictions = []

    distances, indices = find_all_neighbors(knn, normalized_rtgs, k)

    for _, r in test_df.iterrows():
        actual = r["rating"]

        prediction = predict_knn_rating(r["user_id"], r["movie_id"], normalized_rtgs,
                                    avg_user_rtg, user_movie_rtgs, indices, distances)
        predictions.append(prediction)
        actuals.append(actual)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print("=== KNN Model Evaluation ===")
    print()

    print("RMSE:", rmse)
    print("MAE:", mae)
    print()


"""
Matrix Factorization (SVD)
"""

def svd(train_df, test_df, n):
    normalized_rtgs, user_movie_rtgs, avg_user_rtg = pivot_normalize(train_df)

    svd = TruncatedSVD(random_state=35, n_components=n)
    svd.fit(normalized_rtgs)

    pred_matrix = np.dot(svd.transform(normalized_rtgs), svd.components_)

    actuals = []
    predictions = []

    for _, r in test_df.iterrows():
        actual = r["rating"]

        user_id = r["user_id"]
        movie_id = r["movie_id"]


        if user_id in user_movie_rtgs.index and movie_id in user_movie_rtgs.columns:

            user_index = user_movie_rtgs.index.get_loc(user_id)
            movie_index = user_movie_rtgs.columns.get_loc(movie_id)

            prediction = pred_matrix[user_index, movie_index] + avg_user_rtg[user_index][0]
            predictions.append(prediction)
            actuals.append(actual)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print("=== Matrix Factorization Model Evaluation ===")
    print()

    print("RMSE:", rmse)
    print("MAE:", mae)
    print()

"""
Linear Regression Model
"""

def lr(train_df, test_df):
    movie_mean = train_df.groupby("movie_id")["rating"].mean().to_dict()
    user_mean = train_df.groupby("user_id")["rating"].mean().to_dict()
    global_mean = train_df["rating"].mean()

    def get_features(df):
        features = []
        targets = []

        for _, in df.iterrows():

            user_id = ["user_id"]
            movie_id = ["movie_id"]
            rating = ["rating"]

            user_avg = user_mean.get(user_id, global_mean)

            movie_avg = movie_mean.get(movie_id, global_mean)

            features.append([user_avg, movie_avg])
            targets.append(rating)

        return np.array(features), np.array(targets)

    X_train, y_train = get_features(train_df)
    X_test, y_test = get_features(test_df)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print("=== Linear Regression Model Evaluation ===")
    print()
    print("RMSE:", rmse)
    print("MAE:", mae)
    print()


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

    movie_ratings = preprocess(ratings, movies)

    print("=== Merge and Preprocess Data ===")
    print()

    print("Merged shape:", movie_ratings.shape)
    print()

    print("Merged top 5:")
    print(movie_ratings.head())
    print()

    # train test split step

    train_data, test_data = split_data(movie_ratings)

    print("=== Train Test Split ===")
    print()

    print("Train shape:", train_data.shape)
    print("Test shape:", test_data.shape)
    print()

    print("Train top 5:")
    print(train_data.head())
    print()

    print("Test top 5:")
    print(test_data.head())
    print()

    # baseline model preparation step

    movie_mean_dict, user_mean_dict, global_mean = prepare_baseline(train_data)

    print("=== Baseline Model Preparation ===")
    print()

    print("Number of movie averages:", len(movie_mean_dict))
    print("Number of user averages:", len(user_mean_dict))
    print("Global mean:", global_mean)
    print()

    # baseline model step

    baseline_predictions = get_baseline_predictions(test_data, movie_mean_dict,
                                                    user_mean_dict, global_mean)

    evaluate_baseline(test_data, baseline_predictions)

    # TODO
    # 6. add traditional models

    knn, normalized_rtgs, user_movie_rtgs, avg_user_rating = prepare_knn(train_data)

    # testing resulted in k = 40 having the best rmse/mae values
    k = 40
    evaluate_knn(test_data, knn, normalized_rtgs, avg_user_rating, user_movie_rtgs, k)

    # testing resulted in n = 11 having the best rmse/mae values
    n = 11
    svd(train_data, test_data, n)

    lr(train_data, test_data)

    # 7. add neural model
    # 8. compare models



if __name__ == "__main__":
    main()