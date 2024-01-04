import datetime
import yaml

from LSTM import *
from ARIMA import *
from Pipeline import *


log_transform=False
# name_of_sector = 'wig-banki' # będzie się zmieniać
# ticker = 'PKO.WA' # będzie się zmieniać
start_date = datetime(2016, 1, 1)
end_date = datetime(2023, 11, 30)
features_list = ['Close']
lookback = 50
dropout = 0.3
epochs = 100
batch_size = 64


def run_LSTM_based_model(data, data_to_train, data_to_test, name_of_submethod, name_of_sector, ticker, dropout, epochs, batch_size):

    if name_of_submethod=='BatchNormalization':
        dropout = 0.0

    Y_train_predictions, Y_test_predictions = run_predict_prices_LSTM(
        log_transform = log_transform, 
        name_of_submethod = name_of_submethod,
        name_of_sector = name_of_sector,
        ticker = ticker,
        data = data,
        data_to_train = data_to_train,
        data_to_test = data_to_test,
        num_features = 1,
        lookback = lookback,
        horizon = 1,
        learning_rate = 1e-3,
        dropout=dropout,
        epochs = epochs, 
        batch_size = batch_size)
    

    if log_transform:
        model_save_dir = save_inverse_log_results(log_transform, ticker, 'LSTM', name_of_submethod, name_of_sector,
                             data, data_to_train, data_to_test, Y_train_predictions, Y_test_predictions, epochs, dropout)

        plot_LSTM_results(Y_train_predictions, Y_test_predictions, 
                        data_to_train, data_to_test, ticker, model_save_dir, lookback, plot_inverse_prediction=True)


def run_ARIMA_based_model(data_to_train, data_to_test, name_of_submethod, name_of_sector, ticker):
    arima_model, params, prediction = ARIMA_model_actions(data_to_train, data_to_test, lookback, name_of_submethod)

    model_save_dir = save_ARIMA_results(ticker, name_of_submethod, name_of_sector, 
                                        data_to_train, data_to_test, arima_model, params, prediction)
    
    plot_ARIMA_results(data_to_test, params, prediction, lookback, ticker, model_save_dir)



if __name__ == '__main__':
    with open('config.yaml') as f:
        ticker_dict = yaml.safe_load(f)
    
    for name_of_sector in ticker_dict:
        for ticker in ticker_dict[name_of_sector]:
            print(ticker)


            data, data_to_train, data_to_test = split_data(
                    ticker, start_date, end_date, features_list
            )

            run_LSTM_based_model(data, data_to_train, data_to_test, 'dropout_0_3', name_of_sector, ticker, dropout, epochs, batch_size)
            # define name_of_submethod as BatchNormalization to add BatchNormalization to model
            run_LSTM_based_model(data, data_to_train, data_to_test, 'BatchNormalization', name_of_sector, ticker, dropout, epochs, batch_size)
            run_ARIMA_based_model(data_to_train, data_to_test, 'rolling_window', name_of_sector, ticker)
            run_ARIMA_based_model(data_to_train, data_to_test, 'walk_forward_validation', name_of_sector, ticker)

