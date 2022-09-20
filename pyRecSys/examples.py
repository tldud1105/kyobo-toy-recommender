# Other
import os
import time

# Data manipulation
import pandas as pd

# Modeling
from pyRecSys import BaselineModel, KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error

# Movie data found here https://grouplens.org/datasets/movielens/
DATA_PATH = 'data'
FILE_NAME = 'movielens/ml-25m/ratings.csv'
ratings = pd.read_csv(os.path.join(DATA_PATH, FILE_NAME))
ratings = ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id'})
X = ratings[['user_id', 'item_id']]
y = ratings['rating']

# Prepare data without update
# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.1)

# Prepare data with update
(
    X_train,
    y_train,
    X_update,
    y_update,
    X_test,
    y_test
 ) = train_update_test_split(ratings, frac_new_users=0.2)


def execution(method: str = 'sgd'):
    """
    Args:
        method: {str} -- Method to estimate parameters. Can be one of 'sgd', 'als', 'mf_linear' (default: {'sgd'})
    """
    if method == 'sgd':
        # Initial training -- SGD
        baseline_model = BaselineModel(method='sgd', n_epochs=50, reg=0.05, lr=0.01, verbose=1)
        baseline_model.fit(X_train, y_train)

        # Update model with new users
        baseline_model.update_users(X_update, y_update, n_epochs=20, reg=0.05, lr=0.01, verbose=1)

        # Prediction
        y_pred = baseline_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f'\nTest RMSE: {rmse:.4f}')

        # Get recommendations
        user = 200
        items_known = X_train.query("user_id == @user")["item_id"]
        print(baseline_model.recommend(user=user, items_known=items_known))
    elif method == 'als':
        # Initial training -- ALS
        baseline_model = BaselineModel(method='als', n_epochs=200, reg=0.1, verbose=1)
        baseline_model.fit(X_train, y_train)

        # Update model with new users
        baseline_model.update_users(X_update, y_update, n_epochs=50, reg=0.1, verbose=1)

        # Prediction
        y_pred = baseline_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f'\nTest RMSE: {rmse:.4f}')

        # Get recommendations
        user = 200
        items_known = X_train.query("user_id == @user")["item_id"]
        print(baseline_model.recommend(user=user, items_known=items_known))
    elif method == 'mf_linear':
        # Initial training
        mf_linear_model = KernelMF(n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005)
        mf_linear_model.fit(X_train, y_train)

        # Update model with new users
        mf_linear_model.update_users(X_update, y_update, lr=0.001, n_epochs=20, verbose=1)
        y_pred = mf_linear_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"\nTest RMSE: {rmse:.4f}")

        # Get recommendations
        user = 200
        items_known = X_train.query("user_id == @user")["item_id"]
        print(mf_linear_model.recommend(user=user, items_known=items_known))
    elif method == 'mf_sigmoid':
        # Initial training
        mf_sigmoid_model = KernelMF(n_epochs=30, n_factors=100, kernel='sigmoid', verbose=1, lr=0.01, reg=0.05)
        mf_sigmoid_model.fit(X_train, y_train)

        # Update model with new users
        mf_sigmoid_model.update_users(X_update, y_update, lr=0.001, n_epochs=20, verbose=1)
        y_pred = mf_sigmoid_model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"\nTest RMSE: {rmse:.4f}")

        # Get recommendations
        user = 200
        items_known = X_train.query("user_id == @user")["item_id"]
        print(mf_sigmoid_model.recommend(user=user, items_known=items_known))


if __name__ == "__main__":
    start_time = time.time()
    #execution(method='sgd')
    #execution(method='als')
    execution(method='mf_linear')
    #execution(method='mf_sigmoid')
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))