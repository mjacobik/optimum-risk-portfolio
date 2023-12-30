import datetime, sys, os
import matplotlib.pyplot as plt
import numpy as np
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

from Pipeline.save_results import *

def _walk_forward_validation(data_to_train, data_to_test, params, lookback):
    prediction = []
    data = data_to_train.values
    for t in data_to_test.values:
        model=ARIMA(data,order=params)
        model=model.fit()
        output = model.forecast()
        yhat = output[0]
        prediction.append(yhat)
        data = np.append(data, t)
    
    return prediction[lookback:]


def _sliding_window_method(data_to_test, params, lookback):
    prediction = []
    data = data_to_test.values[:lookback]
    for t in data_to_test.values[lookback:]:
        model=ARIMA(data,order=params)
        model=model.fit()
        output = model.forecast()
        yhat = output[0]
        prediction.append(yhat)
        data = np.append(data, t)
        data = np.delete(data, 0)
    
    return prediction


def ARIMA_model_actions(data_to_train, data_to_test, lookback, name_of_submethod):
    # fit params
    fit = auto_arima(data_to_train['Close'], trace=False)
    params = fit.get_params().get("order")

    if name_of_submethod=='rolling_window':
        prediction = _sliding_window_method(data_to_test, params, lookback)
    else:
        prediction = _walk_forward_validation(data_to_train, data_to_test, params, lookback)

    return fit, params, prediction


def plot_ARIMA_results(test, params, prediction, lookback, ticker, model_save_dir):

    plt.plot(test, label='Dane rzeczywiste testowe')
    plt.plot(test.index[lookback:], prediction, label='Predykcja')
    plt.title(f"Predykcja cen akcji spółki {ticker} przy użyciu modelu ARIMA{params}")
    plt.legend()

    plt.savefig(os.path.join(model_save_dir, "Figures", ticker + ".png"))
    plt.close()

