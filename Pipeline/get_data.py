import datetime
from typing import List
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


def log_transform():
    pass
