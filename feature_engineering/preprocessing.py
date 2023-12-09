__all__ = [
    'get_data_by_ticker',
    'get_stock_data',
    'perform_scaling',
    # 'scale_train_data',
    '_adjust_X_train_data',
    'prepare_data_for_model'
]

import datetime
from typing import List
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import yfinance as yfin


def get_data_by_ticker(ticker: str, start_date: datetime, end_date: datetime, features: List) -> pd.DataFrame:
    """
    Scrap data of one stock specyfying how many features
    """
    stock_data = yfin.Ticker(ticker)
    stock_price_df = stock_data.history(start=start_date, end=end_date)

    return stock_price_df.loc[:, features]


def get_stock_data(tickers: List, start_date: datetime, end_date: datetime, target_variable: List[str]):
    """
    Scrap one feature of multiple stocks from one sector
    """
    to_get_index = get_data_by_ticker(tickers[0], start_date, end_date, target_variable)
    get_index = pd.to_datetime(to_get_index.index)
    feature_df = pd.DataFrame(index=get_index, columns=tickers)
    
    for ticker in tickers:
        stock_price_df = get_data_by_ticker(ticker, start_date, end_date, target_variable)
        feature_df[ticker] = stock_price_df[' '.join(target_variable)]

    return feature_df


def perform_scaling(series_data: pd.Series) -> pd.Series:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array(series_data).reshape(-1, 1))

    return scaler.transform(np.array(series_data).reshape(-1, 1)), scaler


# def scale_train_data(stock_data: pd.DataFrame, name_feature_to_predict: str) -> pd.DataFrame:
#     """
#     Scale whole dataset - without sense
#     """
#     get_index = pd.to_datetime(stock_data.index)
#     normalized_data = pd.DataFrame(index=get_index, columns=stock_data.columns)
#     for column in normalized_data.columns:
#         if str(column)==name_feature_to_predict:
#             normalized_data[str(column)], scaler = perform_scaling(stock_data.loc[:, str(column)])
#             return_scaler = scaler
#         else:
#             normalized_data[str(column)], scaler = perform_scaling(stock_data.loc[:, str(column)])

#     return normalized_data, return_scaler


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

