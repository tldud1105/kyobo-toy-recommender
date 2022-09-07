import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class SVD:
    def __init__(self, df):
        self.df = df

    def split_train_test(self):
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=0)
        return train_df, test_df

    def preprocessing(self, train_df):

        # make sparse matrix
        sparse_matrix = train_df.groupby('itemId').apply(
            lambda x: pd.Series(x['rating'].values, index=x['userId'])).unstack()
        sparse_matrix.index.name = 'itemId'

        # fill sparse matrix with average of item ratings
        sparse_matrix_withItem = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)

        # fill sparse matrix with average of user ratings
        sparse_matrix_withUser = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)
        return sparse_matrix_withItem, sparse_matrix_withUser

    def get_svd(self, s_matrix, k=300):

        # s_matrix : 2D array
        # u : left singular vector (2D array)
        # s : singular value of sparse matirx (1D array)
        # vh : right singular vector (2D array)
        u, s, vh = np.linalg.svd(s_matrix.transpose())
        S = s[:k] * np.identity(k, np.float64)
        T = u[:, :k]
        Dt = vh[:k, :]

        item_factors = np.transpose(np.matmul(S, Dt))
        user_factors = np.transpose(T)

        return item_factors, user_factors

    def prediction(self, sparse_matrix_withItem, sparse_matrix_withUser):

        # with item
        item_factors, user_factors = self.get_svd(sparse_matrix_withItem)
        prediction_withItem = pd.DataFrame(np.matmul(item_factors, user_factors),
                                                  columns=sparse_matrix_withItem.columns.values,
                                                  index=sparse_matrix_withItem.index.values)
        # with user
        prediction_withItem = prediction_withItem.transpose()
        item_factors, user_factors = self.get_svd(sparse_matrix_withUser)
        prediction_withUser = pd.DataFrame(np.matmul(item_factors, user_factors),
                                                  columns=sparse_matrix_withUser.columns.values,
                                                  index=sparse_matrix_withUser.index.values)
        prediction_withUser = prediction_withUser.transpose()

        return prediction_withItem, prediction_withUser

    def evaluate(self, test_df, prediction_withItem, prediction_withUser):

        # extract intersection IDs between 'test data' and 'prediction data'
        groups_with_item_ids = test_df.groupby(by='itemId')
        groups_with_user_ids = test_df.groupby(by='userId')

        # for item
        intersection_item_ids = sorted(
            list(set(list(prediction_withItem.columns)).intersection(set(list(groups_with_item_ids.indices.keys())))))
        intersection_user_ids = sorted(
            list(set(list(prediction_withItem.index)).intersection(set(groups_with_user_ids.indices.keys()))))
        compressed_prediction_withItem = prediction_withItem.loc[intersection_user_ids][intersection_item_ids]
        # for user
        intersection_item_ids = sorted(
            list(set(list(prediction_withUser.columns)).intersection(set(list(groups_with_item_ids.indices.keys())))))
        intersection_user_ids = sorted(
            list(set(list(prediction_withUser.index)).intersection(set(groups_with_user_ids.indices.keys()))))
        compressed_prediction_withUser = prediction_withUser.loc[intersection_user_ids][intersection_item_ids]

        # calculation RMSE for test data
        # with item
        rmse_withItem = pd.DataFrame(columns=['rmse'])
        print('evaluate for ')
        for userId, group in tqdm(groups_with_user_ids):
            if userId in intersection_user_ids:
                pred_ratings = compressed_prediction_withItem.loc[userId][
                    compressed_prediction_withItem.loc[userId].index.intersection(list(group['itemId'].values))]
                pred_ratings = pred_ratings.to_frame(name='rating').reset_index().rename(
                    columns={'index': 'itemId', 'rating': 'pred_rating'})
                actual_ratings = group[['rating', 'itemId']].rename(columns={'rating': 'actual_rating'})

                final_withItem = pd.merge(actual_ratings, pred_ratings, how='inner', on=['itemId'])
                final_withItem = final_withItem.round(4)

                if not final_withItem.empty:
                    rmse = sqrt(mean_squared_error(final_withItem['actual_rating'], final_withItem['pred_rating']))
                    rmse_withItem.loc[userId] = rmse

        # calculation RMSE for test data
        # with user
        rmse_withUser = pd.DataFrame(columns=['rmse'])
        for userId, group in tqdm(groups_with_user_ids):
            if userId in intersection_user_ids:
                pred_ratings = compressed_prediction_withUser.loc[userId][
                    compressed_prediction_withUser.loc[userId].index.intersection(list(group['itemId'].values))]
                pred_ratings = pred_ratings.to_frame(name='rating').reset_index().rename(
                    columns={'index': 'itemId', 'rating': 'pred_rating'})
                actual_ratings = group[['rating', 'itemId']].rename(columns={'rating': 'actual_rating'})

                final_withUser = pd.merge(actual_ratings, pred_ratings, how='inner', on=['itemId'])
                final_withUser = final_withUser.round(4)

                if not final_withUser.empty:
                    rmse = sqrt(mean_squared_error(final_withUser['actual_rating'], final_withUser['pred_rating']))
                    rmse_withUser.loc[userId] = rmse

        return final_withItem, final_withUser, rmse_withItem, rmse_withUser


if __name__ == '__main__':
    DATA_PATH = 'data'
    FILE_NAME = 'movielens/ratings.csv'
    df = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))
    df.columns = ['userId', 'itemId', 'rating', 'timestamp']
    test = SVD(df)
    train_df, test_df = test.split_train_test()
    sparse_matrix_withItem, sparse_matrix_withUser = test.preprocessing(train_df)
    prediction_withItem, prediction_withUser = test.prediction(sparse_matrix_withItem, sparse_matrix_withUser)
    _, _, rmse_withItem, rmse_withUser = test.evaluate(test_df, prediction_withItem, prediction_withUser)