import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from .metrics import mse_score, rmse_score, r2_score

class LinearRegressor():
    """Linear regressor"""

    def __init__(self, method='normal_equation', lr=0.01, epochs=1000):
        assert method in ['normal_equation', 'gradient_descent'], "Method not supported. Supported methods are 'normal_equation' and 'gradient_descent'"
        self.method = method
        if self.method == 'gradient_descent':
            self.lr = lr
            self.epochs = epochs
        else:
            pass
    
    def fit(self, X, y):
        """Fit the model to the data"""

        X = X.to_numpy()
        y = y.to_numpy()

        if self.method == 'normal_equation':
            self._weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

        else:
            # mse_new = np.inf
            self._weights = np.zeros(X.shape[1])
            self.cost_history = [0] * self.epochs
            m = len(y)

            for i in range(self.epochs):
                self._weights = self._weights - (self.lr/m) * np.dot(X.T, np.dot(X, self._weights) - y)
                self.cost_history[i] = mse_score(y, np.dot(X, self._weights))

            #     if (rmse_new > rmse_old):
            #         print("Stopped at iteration {}".format(i))
            #         break
            plt.scatter(range(self.epochs), self.cost_history)
            plt.xlabel('epoch')
            plt.ylabel('mse')

    def predict(self, X):
        """Use the fitted model to predict on data"""

        assert self._weights, "Model needs to be fitted first. Use the fit method"
        X = X.to_numpy()
        y = np.dot(X, self._weights)
        return y
    
    def get_weights(self):
        """Get weights from the fitted model"""

        assert self._weights, "Model needs to be fitted first. Use the fit method"
        return self._weights

    def score(self, X, y, metric='r2'):
        """Score the model"""

        assert metric in ['r2', 'rmse'], "Metric not supported. Supported metrics are 'r2', 'mae' and 'rmse'"
        y_pred = self.predict(X)
        if metric == 'r2':
            score = r2_score(y, y_pred)
        elif metric == 'rmse':
            score = rmse_score(y, y_pred)
        elif metric == 'mae':
            score = mae_score(y, y_pred)

        return score