import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2):
    """Perform a split of the dataframe into train and test sets"""

    assert len(X) == len(y), "X and y lengths don't match"

    msk = np.random.rand(len(X)) > test_size
    X_train = X[msk]
    y_train = y[msk]
    X_test = X[~msk]
    y_test = y[~msk]
    return X_train, y_train, X_test, y_test

def standardize_data(df):
    """Standardize columns of the dataframe, returning a centered dataframe"""

    df_norm = (df - df.mean()) / (df.std())
    return df_norm
