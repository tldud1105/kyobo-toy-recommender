import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data_path = 'data'
file_name = 'movielens/ratings.csv'


def preprocessing():
    # load data
    df = pd.read_csv(os.path.join(data_path, file_name))
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

    # make sparse matrix
    sparse_matrix = train_df.groupby('movieId').apply(
        lambda x: pd.Series(x['rating'].values, index=x['userId'])).unstack()
    sparse_matrix.index.name = 'movieId'

    # fill sparse matrix with average of movie ratings
    sparse_matrix_withmovie = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)

    # fill sparse matrix with average of user ratings
    sparse_matrix_withuser = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)

def get_svd(s_matrix, k=300):
    # s_matrix : 2D array
    # u : left singular vector
    # s : singular value of s_matirx
    # vh : right singular vector
    u, s, vh = np.linalg.svd(s_matrix.transpose())
    S = s[:k] * np.identity(k, np.float)
    T = u[:,:k]
    Dt = vh[:k,:]

    item_factors = np.transpose(np.matmul(S, Dt))
    user_factors = np.transpose(T)

    return item_factors, user_factors


if __name__ == '__main__':
