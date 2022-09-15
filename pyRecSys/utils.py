import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Tuple


def train_update_test_split(
    df: pd.DataFrame, frac_new_users: float
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into 3 parts (train, update, test) for testing performance of model update for new users.
    First, a set of new users is set and all ratings corresponding to all other users is assigned to train set.
    Then, for each new user half of their ratings are stored in update set and half are stored in test set.

    Args:
        df {pd.DataFrame}: Data frame containing columns user_id, item_id, rating
        frac_new_users {float}: Fraction of users to not include in train

    Returns:
        X_train {pd.DataFrame}: Training set user_ids and item_ids for initial model fitting
        y_train {pd.Series}: Corresponding ratings for X_train
        X_update {pd.DataFrame}: Updating Set user_ids and item_ids for model updating. Contains users that are not in X_train
        y_update {pd.Series}: Corresponding ratings for X_update
        X_test {pd.DataFrame}: Testing set user_ids and item_ids for model updating. Contains same users as X_update
        y_test {pd.Series}: Corresponding ratings for X_test
    """
    users = df["user_id"].unique()

    # Users that won't be included in the initial training
    non_train_users = np.random.choice(
        users, size=round(frac_new_users * len(users)), replace=False
    )

    # Initial training matrix
    train_df = df.query("user_id not in @non_train_users").sample(
        frac=1, replace=False
    )

    # Update and test sets for non-train model.
    # For each new user split their ratings into two sets, one for update and one for test
    non_train_df = df.query("user_id in @non_train_users")
    update_df, test_df = train_test_split(
        non_train_df, stratify=non_train_df["user_id"], test_size=0.5
    )

    # Split into X and y
    X_train, y_train = (
        train_df[["user_id", "item_id"]],
        train_df["rating"],
    )
    X_update, y_update = (
        update_df[["user_id", "item_id"]],
        update_df["rating"],
    )
    X_test, y_test = (
        test_df[["user_id", "item_id"]],
        test_df["rating"],
    )

    return (
        X_train,
        y_train,
        X_update,
        y_update,
        X_test,
        y_test,
    )

