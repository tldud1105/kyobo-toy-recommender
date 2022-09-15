# Other
import os

# Data manipulation
import pandas as pd


# Load data
DATA_PATH = 'data'
FILE_NAME = 'movielens/ml-25m/ratings.csv'
ratings = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))

DATA_PATH = 'data'
FILE_NAME = 'movielens/ml-25m/links.csv'
links = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))

DATA_PATH = 'data'
FILE_NAME = 'movielens/ml-25m/movies.csv'
movies = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))

DATA_PATH = 'data'
FILE_NAME = 'movielens/ml-25m/tags.csv'
tags = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))

ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
X = ratings[['user_id', 'item_id']]
y = ratings['rating']

print(ratings.head(10))

