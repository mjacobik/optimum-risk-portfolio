import datetime
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# # setting path
# sys.path.append(os.path.join('..', 'Pipeline'))

from Pipeline.save_results import *


def _walk_forward_validation(data_to_train, data_to_test, params):
    prediction = []
    data = data_to_train.values
    for t in data_to_test.values:
        model=ARIMA(data,order=params)
        model=model.fit()
        output = model.forecast()
        yhat = output[0]
        prediction.append(yhat)
        data = np.append(data, t)
    
    return prediction


def ARIMA_model_actions(data, data_to_train, data_to_test):
    # fit params
    fit = auto_arima(data_to_train['Close'], trace=False)
    params = fit.get_params().get("order")

    prediction = _walk_forward_validation(data_to_train, data_to_test, params)

    return data, data_to_train, data_to_test, params, prediction


def plot_ARIMA_results(test, prediction, lookback, ticker, params):
    plt.plot(test.iloc[lookback:, :], label='Dane rzeczywiste')
    plt.plot(test.index[lookback:], prediction[lookback:], label='Predykcja')
    plt.title(f"Predykcja cen akcji spółki {ticker} przy użyciu modelu ARIMA{params}")
    plt.legend()
    plt.show()

    # plt.savefig(os.path.join(model_save_dir, "Figures", ticker + ".png"))
    # plt.close()


if __name__ == '__main__':
    data_to_train, data_to_test, params = ARIMA_model_actions('PKO.WA', datetime(2016, 1, 1), datetime(2023, 11, 30), ['Close'])
    _walk_forward_validation(data_to_train, data_to_test, params)
