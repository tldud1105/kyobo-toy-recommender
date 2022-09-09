import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class SVD:
    def __init__(self, df):
        self.df = df

    def split_train_test(self):
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=1234)
        return train_df, test_df

    def mk_sparse_matrix(self, train_df):
        # make sparse matrix
        sparse_matrix = train_df.groupby('itemId').apply(
            lambda x: pd.Series(x['rating'].values, index=x['userId'])).unstack()
        return sparse_matrix

    def fill_with_avgItem(self, sparse_matrix):
        # fill sparse matrix with average of item ratings
        matrix_with_avgItem = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)
        return matrix_with_avgItem

    def fill_with_avgUser(self, sparse_matrix):
        # fill sparse matrix with average of user ratings
        matrix_with_avgUser = sparse_matrix.apply(lambda x: x.fillna(x.mean()), axis=0)
        return matrix_with_avgUser

    def get_svd(self, matrix, k):
        u, s, vh = np.linalg.svd(matrix)
        T = u[:, :k]
        S = s[:k] * np.identity(k)
        Dt = vh[:k, :]
        item_factors = T
        user_factors = np.matmul(S, Dt)
        return item_factors, user_factors

    def prediction(self, k, status):

        train_df, test_df = self.split_train_test()
        sparse_matrix = self.mk_sparse_matrix(train_df)

        if status == 'avgItem':
            matrix_with_avgItem = self.fill_with_avgItem(sparse_matrix)
            item_factors, user_factors = self.get_svd(matrix_with_avgItem, k)
            prediction_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                         columns=matrix_with_avgItem.columns.values,
                                         index=matrix_with_avgItem.index.values)
        elif status == 'avgUser':
            matrix_with_avgUser = self.fill_with_avgUser(sparse_matrix)
            item_factors, user_factors = self.get_svd(matrix_with_avgUser, k)
            prediction_df = pd.DataFrame(np.matmul(item_factors, user_factors),
                                         columns=matrix_with_avgUser.columns.values,
                                         index=matrix_with_avgUser.index.values)

        intersection_item_ids = sorted(
            list(set(prediction_df.index).intersection(set(test_df['itemId']))))
        intersection_user_ids = sorted(
            list(set(prediction_df.columns).intersection(set(test_df['userId']))))
        compressed_prediction_df = prediction_df.loc[intersection_item_ids][
            intersection_user_ids]

        transposed_prediction_df = compressed_prediction_df.transpose()
        final_prediction_df = pd.DataFrame(columns=['userId', 'itemId', 'actual_rating', 'pred_rating'])
        grouped = test_df.groupby(by='userId')
        for userId, group in tqdm(grouped):
            if userId in intersection_user_ids:
                pred_ratings = transposed_prediction_df.loc[userId][
                    transposed_prediction_df.loc[userId].index.intersection(list(group['itemId'].values))]
                pred_ratings = pred_ratings.to_frame(name='rating').reset_index().rename(
                    columns={'index': 'itemId', 'rating': 'pred_rating'})
                actual_ratings = group[['rating', 'itemId']].rename(columns={'rating': 'actual_rating'})
                oneId_df = pd.merge(actual_ratings, pred_ratings, how='inner', on=['itemId'])
                oneId_df = oneId_df.round(4)
                oneId_df['userId'] = userId
                final_prediction_df = pd.concat([final_prediction_df, oneId_df])

        return final_prediction_df


if __name__ == '__main__':
    DATA_PATH = 'data'
    FILE_NAME = 'movielens/ratings.csv'
    df = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))
    df.columns = ['userId', 'itemId', 'rating', 'timestamp']

    # In case of k=100
    # 'with average item ratings' vs 'with average user ratings'
    # You can see that the RMSE of 'with average user ratings' is smaller than that of 'with average item ratings'
    # It is because we used the movie dataset in our analysis.
    # Perhaps the personality of the person was more important factor than the characteristics of each movie.
    test = SVD(df)
    prediction_avgItem = test.prediction(k=100, status='avgItem')
    prediction_avgUser = test.prediction(k=100, status='avgUser')
    print(
        f"RMSE: {sqrt(mean_squared_error(prediction_avgItem['actual_rating'].values, prediction_avgItem['pred_rating'].values))}")
    print(
        f"RMSE: {sqrt(mean_squared_error(prediction_avgUser['actual_rating'].values, prediction_avgUser['pred_rating'].values))}")

    # grid search for k
    def find_best_k(start_k, stop_k, step_k, status):
        k_candidates = np.arange(start_k, stop_k, step_k)
        rmse_df = pd.DataFrame(columns=['rmse'], index=k_candidates)
        for k in tqdm(k_candidates):
            test = SVD(df)
            if status == 'avgItem':
                prediction_df = test.prediction(k=k, status='avgItem')
            elif status == 'avgUser':
                prediction_df = test.prediction(k=k, status='avgUser')
            rmse = sqrt(
                mean_squared_error(prediction_df['actual_rating'].values, prediction_df['pred_rating'].values))
            rmse_df.loc[k]['rmse'] = rmse
        return rmse_df

    start_k = 1
    stop_k = 100
    step_k = 5
    rmse_df_avgItem = find_best_k(start_k, stop_k, step_k, status='avgItem')
    rmse_df_avgUser = find_best_k(start_k, stop_k, step_k, status='avgUser')