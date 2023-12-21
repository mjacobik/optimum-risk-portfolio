import datetime

from LSTM import *
from ARIMA import *
from Pipeline import *


def run_ARIMA_based_model():
    pass


if __name__ == '__main__':
    data = get_data_by_ticker('PKO.WA', datetime(2016, 1, 1), datetime(2023, 11, 30), ['Close'])
    data_to_train = data.loc[:'2020']
    data_to_test = data.loc['2023':]

    run_predict_prices_LSTM('wig-banki', 'PKO.WA', data, data_to_train, data_to_test, num_features=1, lookback=50, horizon=1, learning_rate=1e-3)

