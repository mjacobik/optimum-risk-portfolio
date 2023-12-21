import numpy as np
import pandas as pd


def _adjust_X_train_data(X_train_data, lookback):
    tmp = []
    for i in range(0, X_train_data.shape[0]-lookback):
        a = slice(i, i+lookback)
        tmp.append(X_train_data[a, :])
    
    return np.array(tmp)


def prepare_data_for_model(train_data, lookback):
    X_train = _adjust_X_train_data(train_data, lookback)
    Y_train = X_train[:, -1, :]

    return X_train, Y_train

