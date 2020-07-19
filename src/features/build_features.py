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

class StandardScaler():
    """Scaler for data standardization"""

    def __init__(self):
        pass

    def fit(self, df):
        """Fit the scaler to the data"""

        self.mean = df.mean()
        self.std = df.std()

    def transform(self, df):
        """Standardize columns of the dataframe, returning a centered dataframe"""

        return (df - self.mean) / self.std

    def fit_transform(self, df):
        """Fit the scaler to the data and return the data standardized"""
        
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        """"Rescale standardized data to its original form"""

        return df * self.std + self.mean

class MinMaxScaler():
    """Scaler for data normalization"""

    def __init__(self):
        pass

    def fit(self, df):
        """Fit the scaler to the data"""

        self.min = df.min()
        self.max = df.max()

    def transform(self, df):
        """Standardize columns of the dataframe, returning a centered dataframe"""

        return (df - self.min) / (self.max - self.min)

    def fit_transform(self, df):
        """Fit the scaler to the data and return the data standardized"""
        
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, df):
        """"Rescale standardized data to its original form"""

        return df * (self.max - self.min) + self.min

