# Other
import os
import time

# Data manipulation
import pandas as pd

# Modeling
from pyRecSys import BaselineModel, train_update_test_split
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

def execution():
    # Initial training -- SGD
    baseline_model = BaselineModel(method='sgd', n_epochs=100, reg=0.05, lr=0.01, verbose=1)
    baseline_model.fit(X_train, y_train)

    # Update model with new users
    baseline_model.update_users(X_update, y_update, n_epochs=100, reg=0.05, lr=0.01, verbose=1)

    # Prediction
    y_pred = baseline_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f'\nTest RMSE: {rmse:.4f}')

    # Get recommendations
    user = 200
    items_known = X_train.query("user_id == @user")["item_id"]
    baseline_model.recommend(user=user, items_known=items_known)

if __name__ == "__main__":
    execution()
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))