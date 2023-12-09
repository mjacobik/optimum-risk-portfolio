import os, sys
from sklearn.preprocessing import MinMaxScaler

# setting path
sys.path.append(os.path.join('..', 'feature_engineering'))
sys.path.append(os.path.join('..', 'LSTM'))

from preprocessing import *
from model import *
from save_results import *


def _train_LSTM_model(data, num_features, lookback, horizon, learning_rate):
    model = build_model(num_features, lookback, horizon, learning_rate)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_train_data = scaler.fit_transform(data)
    # scaled_train_data, scaler = perform_scaling(data)

    X_train, Y_train = prepare_data_for_model(scaled_train_data, lookback)

    model_history = model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=2)

    return scaler, X_train, Y_train, model_history, model


def make_prediction_using_LSTM(data, model, scaler):
    transformed_predictions = model.predict(data)
    return scaler.inverse_transform(transformed_predictions)


def LSTM_model_actions(ticker, start_date, end_date, features,
                  num_features, lookback, horizon, learning_rate):
    
    data = get_data_by_ticker(ticker, start_date, end_date, features)
    data_to_train = data.loc[:'2020']
    data_to_test = data.loc['2023':]

    scaler, X_train, Y_train, model_history, model = _train_LSTM_model(
        data_to_train, num_features, lookback, horizon, learning_rate)
    
    Y_train_predictions = make_prediction_using_LSTM(X_train, model, scaler)

    scaled_test_data = scaler.transform(data_to_test)
    X_test, Y_test = prepare_data_for_model(scaled_test_data, lookback)

    Y_test_predictions = make_prediction_using_LSTM(X_test, model, scaler)

    return {'data': data,
            'data_to_train': data_to_train,
            'data_to_test': data_to_test,
            'scaler': scaler,
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


def run_predict_prices_LSTM(list_of_tickers, start_date, end_date, features,
                            num_features, lookback, horizon, learning_rate):
    
    for ticker in list_of_tickers:
        data, data_to_train, data_to_test, scaler, model_history, model, \
        X_train, Y_train, Y_train_predictions, \
        X_test, Y_test, Y_test_predictions = LSTM_model_actions(
            ticker, start_date, end_date, features,
            num_features, lookback, horizon, learning_rate).values()
    
        model_save_dir = save_LSTM_results(
            ticker, "LSTM", data, scaler, model, model_history,
            X_train, Y_train, Y_train_predictions, X_test, 
            Y_test, Y_test_predictions)
        
        _plot_LSTM_results(Y_train_predictions, Y_test_predictions, 
                           data_to_train, data_to_test, ticker, model_save_dir, lookback)


if __name__ == '__main__':
    run_predict_prices_LSTM(['PKO.WA', 'PEO.WA', 'SPL.WA', 'ING.WA', 'MBK.WA'], 
                            datetime(2016, 1, 1), datetime(2023, 11, 30),
                            ['Close'], num_features=1, lookback=50, horizon=1, learning_rate=1e-3)

