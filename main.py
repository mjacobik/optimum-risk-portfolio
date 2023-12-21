import datetime

from LSTM import *
from ARIMA import *
from Pipeline import *

def run_LSTM_based_model():
    pass

def run_ARIMA_based_model():
    pass


if __name__ == '__main__':
    data = get_data()
    data_to_train = data.loc[:'2020']
    data_to_test = data.loc['2023':]

    run_LSTM_based_model()