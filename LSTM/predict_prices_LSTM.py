import datetime
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from sklearn.preprocessing import MinMaxScaler

# # setting path
# sys.path.append(os.path.join('..', 'Pipeline'))

from .LSTM_preprocessing import *
from .model import *
from Pipeline.save_results import *


def _train_LSTM_model(data, num_features, lookback, horizon, learning_rate):
    model = build_model(num_features, lookback, horizon, learning_rate)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train_data = scaler.fit_transform(data)

    X_train, Y_train = prepare_data_for_model(scaled_train_data, lookback)

    model_history = model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=2)

    return scaler, X_train, Y_train, model_history, model


def _make_prediction_using_LSTM(data, model, scaler):
    transformed_predictions = model.predict(data)
    return scaler.inverse_transform(transformed_predictions)


def LSTM_model_actions(data_to_train, data_to_test, num_features, lookback, horizon, learning_rate):
    # train LSTM model
    scaler, X_train, Y_train, model_history, model = _train_LSTM_model(
        data_to_train, num_features, lookback, horizon, learning_rate)
    
    Y_train_predictions = _make_prediction_using_LSTM(X_train, model, scaler)

    # adjust test data and make final predictions
    scaled_test_data = scaler.transform(data_to_test)
    X_test, Y_test = prepare_data_for_model(scaled_test_data, lookback)

    Y_test_predictions = _make_prediction_using_LSTM(X_test, model, scaler)

    return {'scaler': scaler,
            'model_history': model_history,
            'model' : model,
            'X_train': X_train,
            'Y_train': Y_train,
            'Y_train_predictions': Y_train_predictions,
            'X_test': X_test,
            'Y_test': Y_test,
            'Y_test_predictions': Y_test_predictions}


def _prepare_data_for_plot_LSTM_results(data_to_train, data_to_test, Y_train_predictions, Y_test_predictions, lookback):
    trainPredictPlot = np.empty_like(data_to_train)
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[lookback:(len(data_to_train)), :] = Y_train_predictions

    testPredictPlot = np.empty_like(data_to_test)
    testPredictPlot[:,:] = np.nan
    testPredictPlot[lookback:(len(data_to_test)), :] = Y_test_predictions

    return trainPredictPlot, testPredictPlot


def _plot_LSTM_results(Y_train_predictions, Y_test_predictions, 
                       data_to_train, data_to_test, ticker, model_save_dir, lookback=50):
    
    trainPredictPlot, testPredictPlot = _prepare_data_for_plot_LSTM_results(
        data_to_train, data_to_test, Y_train_predictions, Y_test_predictions, lookback)
    
    plt.plot(data_to_train, label='Dane rzeczywiste')
    plt.plot(data_to_train.index, trainPredictPlot, label='Predykcja')
    plt.plot(data_to_test, label='Dane rzeczywiste testowe')
    plt.plot(data_to_test.index, testPredictPlot, label='Predykcja na zbiorze testowym')
    plt.axvline(x=data_to_train.index[-1], c='r', linestyle='--')
    plt.title(f"Predykcja cen akcji spółki {ticker} przy użyciu modelu LSTM")
    plt.legend()

    plt.savefig(os.path.join(model_save_dir, "Figures", ticker + ".png"))
    plt.close()

    plt.plot(data_to_test, label='Dane rzeczywiste testowe')
    plt.plot(data_to_test.index, testPredictPlot, label='Predykcja')
    plt.title(f"Predykcja cen akcji spółki {ticker} przy użyciu modelu LSTM")
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, "Figures", "test_" + ticker + ".png"))
    plt.close()



def run_predict_prices_LSTM(name_of_sector, ticker, data, data_to_train, data_to_test,
                            num_features, lookback, horizon, learning_rate):
    
    scaler, model_history, model, X_train, Y_train, Y_train_predictions, \
    X_test, Y_test, Y_test_predictions = LSTM_model_actions(
        data_to_train, data_to_test, num_features, lookback, horizon, learning_rate).values()

    model_save_dir = save_LSTM_results(
        ticker, "LSTM", name_of_sector, data, scaler, model, model_history,
        X_train, Y_train, Y_train_predictions, X_test, 
        Y_test, Y_test_predictions)
    
    _plot_LSTM_results(Y_train_predictions, Y_test_predictions, 
                        data_to_train, data_to_test, ticker, model_save_dir, lookback)


# zrobić wczytywanie listy tych tickerów z yamla i uruchomić LSTM od razu dla 45 featerów
# zrobić więcej argumentów w tej funkcji run_predict_prices_LSTM - zrobić porównanie dropoutów
# kazać mu predykować więcej niż jeden dzień w przód i zobaczyć jak sobie radzi