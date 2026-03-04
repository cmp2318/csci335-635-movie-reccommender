import pandas as pd

# starting main, just prints the basic elements from the u.data file

def main():

    # load dataset into pandas
    ratings = pd.read_csv("../data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

    # prints the first 5 rows in a table
    print(ratings.head())

    print("\nTotal Ratings:", len(ratings))
    print("Total Users:", ratings["user_id"].nunique())
    print("Total Movies", ratings["item_id"].nunique())


if __name__ == "__main__":
    main()