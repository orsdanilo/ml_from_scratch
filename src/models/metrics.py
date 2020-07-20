import numpy as np
import math

def r2_score(y_true, y_pred):
    """Calculate the coefficient of determination"""

    # ybar = numpy.sum(y)/len(y) 
    ssreg = np.sum((y_pred - y_true.mean()) ** 2)
    sstot = np.sum((y_true - y_true.mean()) ** 2) 
    r2 = ssreg / sstot
    return r2

def rmse_score(y_true, y_pred):
    """Calculate the root mean squared error"""

    rmse = math.sqrt(np.sum(y_pred - y_true) ** 2 / y_true.size)
    return rmse

def mse_score(y_true, y_pred):
    """Calculate the mean squared error"""

    mse = np.sum(y_pred - y_true) ** 2 / y_true.size
    return mse

def mae_score(y_true, y_pred):
    """Calculate the mean absolute error"""

    mae = np.sum(np.abs(y_pred - y_true) ** 2) / y_true.size
    return mae

def accuracy(y_true, y_pred):
    """Calculate the accuracy"""
    
    acc = (y_true == y_pred).mean()
    return acc

def recall(y_true, y_pred):
    """Calculate recall metric"""
    
    dict_recall = {}
    for label in np.unique(y_pred):
        tp = len(y_pred[(y_pred == label) & (y_pred == y_true)])
        tp_fn = len(y_pred[y_true == label])
        dict_recall[label] = tp / tp_fn
        
    return dict_recall
