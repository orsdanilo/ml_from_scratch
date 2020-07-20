import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from .metrics import accuracy, recall
from ..features.build_features import StandardScaler, MinMaxScaler

class LogisticRegressor():
    """Logistic regressor"""

    def __init__(self, method='max_likelihood', normalize=False, lr=0.01, epochs=1000, add_intercept=False, threshold=0.5):
        assert method in ['max_likelihood', 'cross_entropy'], "Method not supported. Supported methods are 'max_likelihood' and 'cross_entropy'"
        self.method = method
        self.normalize = normalize
        self.add_intercept = add_intercept
        self.threshold = threshold
        self._weights = None
        self.lr = lr
        self.epochs = epochs

        if self.normalize:
            self._feature_scaler = MinMaxScaler()
            
    def _sigmoid(self, z): 
        return 1/(1+np.exp(-z))
    
    def _cross_entropy(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def _log_likelihood(self, z, y):
        ll = np.sum(y*z - np.log(1 + np.exp(z)) )
        return ll
    
    def fit(self, X, y):
        """Fit the model to the data"""
        
        if self.normalize:
            X = self._feature_scaler.fit_transform(X)
          
        X = X.to_numpy()
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.to_numpy()
        
        self._weights = np.zeros(X.shape[1])
        self.cost_history = [0] * self.epochs
        
        if self.method == 'max_likelihood':   
            for i in range(self.epochs):
                z = np.dot(X, self._weights)
                h = self._sigmoid(z)
                grad = np.dot(X.T, y - h)
                self._weights = self._weights + self.lr * grad
                self.cost_history[i] = self._log_likelihood(z, y)
            
        else:           
            for i in range(self.epochs):
                z = np.dot(X, self._weights)
                h = self._sigmoid(z)
                grad = np.dot(X.T, (h - y)) / y.shape[0]
                self._weights = self._weights - self.lr * grad
                self.cost_history[i] = self._cross_entropy(h, y)
                
        plt.scatter(range(self.epochs), self.cost_history)
        plt.xlabel('epoch')
        plt.ylabel(self.method)

    def predict(self, X):
        """Use the fitted model to predict on data"""

        assert self._weights is not None, "Model needs to be fitted first. Use the fit method"

        if self.normalize:
            X = self._feature_scaler.transform(X)

        X = X.to_numpy()
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_prob = np.dot(X, self._weights)

        y_pred = np.where(y_prob >=0.5, 1, 0)

        return y_pred, np.round(y_prob, 2)
    
    def get_weights(self):
        """Get weights from the fitted model"""

        assert self._weights is not None, "Model needs to be fitted first. Use the fit method"
        return self._weights

    def score(self, X, y, metric='r2'):
        """Score the model"""

        assert metric in ['accuracy', 'recall', 'precision'], "Metric not supported. Supported metrics are 'accuracy', 'recall' and 'precision'"

        y_pred, prob = self.predict(X)    
        
        if metric == 'accuracy':
            score = accuracy(y, y_pred)
        elif metric == 'recall':
            score = recall(y, y_pred)
        elif metric == 'precision':
            raise ValueError("Precision score is not yet implemented")
            # score = precision(y, y_pred)

        return score