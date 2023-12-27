import datetime
from typing import List
import numpy as np
import pandas as pd

import yfinance as yfin


def get_data_by_ticker(ticker: str, start_date: datetime, end_date: datetime, features: List) -> pd.DataFrame:
    """
    Scrap data of one stock specyfying how many features

    @return: Pandas dataframe with stock market prices
    """
    stock_data = yfin.Ticker(ticker)
    stock_price_df = stock_data.history(start=start_date, end=end_date)

    return stock_price_df.loc[:, features]


def split_data(ticker, start_date, end_date, features_list):
    data = get_data_by_ticker(ticker, start_date, end_date, features_list)
    data_to_train = data.loc[:'2022']
    data_to_test = data.loc['2023':]

    return data, data_to_train, data_to_test


def transform_and_split_data(data):
    pct_returns = data.pct_change().dropna()
    transformed_data = np.log(1+pct_returns)

    transformed_data_to_train = transformed_data.loc[:'2022']
    transformed_data_to_test = transformed_data.loc['2023':]

    return transformed_data, transformed_data_to_train, transformed_data_to_test

