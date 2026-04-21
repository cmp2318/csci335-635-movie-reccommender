import pandas as pd
import matplotlib.pyplot as plt

import recommender
import neural_model

models = []
rmse_values = []
mae_values = []


"""
adds the model, RMSE, and MAE values to its respective list for comparison
"""

def add_values(model, rmse, mae):
    models.append(model)
    rmse_values.append(rmse)
    mae_values.append(mae)


# starting main, just prints the basic elements from the u.data file

def main():

    # load dataset into pandas
    ratings = pd.read_csv("../data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    # prints the first 5 rows in a table
    print(ratings.head())

    print("\nTotal Ratings:", len(ratings))
    print("Total Users:", ratings["user_id"].nunique())
    print("Total Movies", ratings["item_id"].nunique())

    bl_rmse, bl_mae, knn_rmse, knn_mae, svd_rmse, svd_mae, lr_rmse, lr_mae = recommender.main()
    nm_rmse, nm_mae = neural_model.main()


    # ============================================================
    # Model Comparison
    # ============================================================

    add_values("Baseline", bl_rmse, bl_mae)
    add_values("KNN", knn_rmse, knn_mae)
    add_values("SVD", svd_rmse, svd_mae)
    add_values("Linear Regression", lr_rmse, lr_mae)
    add_values("Neural Network", nm_rmse, nm_mae)

    print(models)
    print(rmse_values)
    print(mae_values)

    plt.bar(models, rmse_values)
    plt.ylim(0.9, 1.05)
    plt.title('RMSE Values')
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.show()

    plt.bar(models, mae_values)
    plt.ylim(0.7, .85)
    plt.title('MAE Values')
    plt.xlabel('Models')
    plt.ylabel('MAE')
    plt.show()


if __name__ == "__main__":
    main()