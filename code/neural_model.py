"""
CSCI 335/635
Movie Recommendation Project

File: neural_model.py

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
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
creates one merged dataframe for the neural network by
combining the ratings data with movie genre data and
user data, each row represents one user rating along
with the extra features tied to that movie and user
"""
def process_neural_data(ratings_df, movies_df, users_df):

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movie_features = movies_df[["movie_id", "title"] + genre_cols]

    neural_df = ratings_df.merge(
        movie_features,
        on="movie_id",
        how="left"
    )

    neural_df = neural_df.merge(
        users_df,
        on="user_id",
        how="left"
    )

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

def prepare_neural_data(train_data, test_data):

    genre_cols = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    train_data = train_data.copy()
    test_data = test_data.copy()

    # get unique ids from training data that will be encoded
    unique_users = sorted(train_data["user_id"].unique())
    unique_movies = sorted(train_data["movie_id"].unique())
    unique_occupations = sorted(train_data["occupation"].unique())

    user_to_index = {}

    # encode user ids to small integers for index access
    for index, user_id in enumerate(unique_users):
        user_to_index[user_id] = index

    movie_to_index = {}

    # same for movies
    for index, movie_id in enumerate(unique_movies):
        movie_to_index[movie_id] = index

    occupation_to_index = {}

    # same for occupations
    for index, occupation in enumerate(unique_occupations):
        occupation_to_index[occupation] = index

    # only keep test rows where the user and movie were seen in training
    test_data = test_data[
        test_data["user_id"].isin(user_to_index) &
        test_data["movie_id"].isin(movie_to_index)
    ].copy()

    # turn the original user ids into index values for embedding
    train_data["user_idx"] = train_data["user_id"].map(user_to_index)
    test_data["user_idx"] = test_data["user_id"].map(user_to_index)

    # turn the original movie ids into index values
    train_data["movie_idx"] = train_data["movie_id"].map(movie_to_index)
    test_data["movie_idx"] = test_data["movie_id"].map(movie_to_index)

    # gender encoding, M = 0, F = 1
    train_data["gender_code"] = train_data["gender"].map({"M": 0, "F": 1})
    test_data["gender_code"] = test_data["gender"].map({"M": 0, "F": 1})

    # convert occupation text into numeric codes
    train_data["occupation_code"] = train_data["occupation"].map(occupation_to_index)
    test_data["occupation_code"] = test_data["occupation"].map(occupation_to_index)

    # scale age to a 0 to 1 range
    min_age = train_data["age"].min()
    max_age = train_data["age"].max()

    train_data["age_scaled"] = (train_data["age"] - min_age) / (max_age - min_age)

    test_data["age_scaled"] = (test_data["age"] - min_age) / (max_age - min_age)

    # inputs that will be passed into the model
    X_train_user = train_data["user_idx"].values
    X_train_movie = train_data["movie_idx"].values
    X_train_genres = train_data[genre_cols].values
    X_train_age = train_data["age_scaled"].values
    X_train_gender = train_data["gender_code"].values
    X_train_occupation = train_data["occupation_code"].values
    y_train = train_data["rating"].values

    X_test_user = test_data["user_idx"].values
    X_test_movie = test_data["movie_idx"].values
    X_test_genres = test_data[genre_cols].values
    X_test_age = test_data["age_scaled"].values
    X_test_gender = test_data["gender_code"].values
    X_test_occupation = test_data["occupation_code"].values
    y_test = test_data["rating"].values

    neural_data = {
        "X_train_user": X_train_user,
        "X_train_movie": X_train_movie,
        "X_train_genres": X_train_genres,
        "X_train_age": X_train_age,
        "X_train_gender": X_train_gender,
        "X_train_occupation": X_train_occupation,
        "y_train": y_train,
        "X_test_user": X_test_user,
        "X_test_movie": X_test_movie,
        "X_test_genres": X_test_genres,
        "X_test_age": X_test_age,
        "X_test_gender": X_test_gender,
        "X_test_occupation": X_test_occupation,
        "y_test": y_test,
        "num_users": len(unique_users),
        "num_movies": len(unique_movies),
        "num_genres": len(genre_cols),
        "num_occupations": len(unique_occupations)
    }

    return neural_data


# ============================================================
# Neural Network Model
# ============================================================

"""
builds a neural network that uses user, movie, and
occupation embeddings along with genre, age, and
gender inputs to predict movie ratings
"""

def build_neural_model(num_users, num_movies, num_occupations, num_genres):

    # all inputs are information the model will receive
    user_input = Input(shape=(1,), name="user_input")
    movie_input = Input(shape=(1,), name="movie_input")
    occupation_input = Input(shape=(1,), name="occupation_input")

    genre_input = Input(shape=(num_genres,), name="genre_input")
    age_input = Input(shape=(1,), name="age_input")
    gender_input = Input(shape=(1,), name="gender_input")

    # allows the model to learn for each user, movie, and occupation
    user_embedding = Embedding(input_dim=num_users, output_dim=16)(user_input)
    movie_embedding = Embedding(input_dim=num_movies, output_dim=16)(movie_input)
    occupation_embedding = Embedding(input_dim=num_occupations, output_dim=8)(occupation_input)

    # flatten the embedding output into normal vector
    user_vector = Flatten()(user_embedding)
    movie_vector = Flatten()(movie_embedding)
    occupation_vector = Flatten()(occupation_embedding)

    # combine all learned vectors and inputs into one vector
    combined = Concatenate()([
        user_vector,
        movie_vector,
        occupation_vector,
        genre_input,
        age_input,
        gender_input
    ])

    # dense layers learn patterns from the combined information
    dense_1 = Dense(64, activation="relu")(combined)
    dense_2 = Dense(32, activation="relu")(dense_1)

    # final output is one predicted rating 
    output = Dense(1, name="rating_output")(dense_2)

    model = Model(
        inputs=[
            user_input,
            movie_input,
            occupation_input,
            genre_input,
            age_input,
            gender_input
        ],
        outputs=output
    )

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model

# ============================================================
# Neural Network Training
# ============================================================

"""
trains the neural network model on the training data
"""

def train_neural_model(model, neural_data):

    history = model.fit(
        [
            neural_data["X_train_user"],
            neural_data["X_train_movie"],
            neural_data["X_train_occupation"],
            neural_data["X_train_genres"],
            neural_data["X_train_age"],
            neural_data["X_train_gender"]
        ],
        neural_data["y_train"],
        epochs=10,
        batch_size=64,
        validation_split=0.10,
        verbose=1
    )

    return history

# ============================================================
# Neural Network Evaluation
# ============================================================

"""
makes predictions on the test set and evaluates the
neural network using rmse and mae
"""

def evaluate_neural_model(model, neural_data):

    predictions = model.predict(
        [
            neural_data["X_test_user"],
            neural_data["X_test_movie"],
            neural_data["X_test_occupation"],
            neural_data["X_test_genres"],
            neural_data["X_test_age"],
            neural_data["X_test_gender"]
        ]
    )

    predictions = predictions.flatten()

    actual = neural_data["y_test"]

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)

    print("Neural Network Evaluation:")
    print()

    print("RMSE:", rmse)
    print("MAE:", mae)
    print()

    return predictions

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

    neural_df = process_neural_data(ratings, movies, users)
    train_data, test_data = split_data(neural_df)

    print("Train shape:", train_data.shape)
    print("Test shape:", test_data.shape)

    print()

    neural_data = prepare_neural_data(train_data, test_data)

    print()

    print("Number of users:", neural_data["num_users"])
    print("Number of movies:", neural_data["num_movies"])
    print("Number of genres:", neural_data["num_genres"])
    print("Number of occupations:", neural_data["num_occupations"])
    print()

    model = build_neural_model(
        neural_data["num_users"],
        neural_data["num_movies"],
        neural_data["num_occupations"],
        neural_data["num_genres"]
    )

    print("Neural Network Model:")
    print()
    model.summary()
    print()

    history = train_neural_model(model, neural_data)

    predictions = evaluate_neural_model(model, neural_data)

if __name__ == "__main__":
    main()