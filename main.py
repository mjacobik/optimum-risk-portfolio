import datetime

from LSTM import *
from ARIMA import *
from Pipeline import *


log_transform=False
name_of_sector = 'wig-budownictwo' # będzie się zmieniać
ticker = 'BDX.WA' # będzie się zmieniać
start_date = datetime(2016, 1, 1)
end_date = datetime(2023, 11, 30)
features_list = ['Close']
lookback = 50
dropout = 0.0
epochs = 200
batch_size = 64


def run_LSTM_based_model(data, data_to_train, data_to_test, name_of_sector, ticker, dropout, epochs, batch_size):
    Y_train_predictions, Y_test_predictions = run_predict_prices_LSTM(
        log_transform = log_transform, 
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
        model_save_dir = save_inverse_log_results(log_transform, ticker, 'LSTM', name_of_sector,
                             data, data_to_train, data_to_test, Y_train_predictions, Y_test_predictions, epochs, batch_size, dropout)

        plot_LSTM_results(Y_train_predictions, Y_test_predictions, 
                        data_to_train, data_to_test, ticker, model_save_dir, lookback, plot_inverse_prediction=True)



def run_ARIMA_based_model():
    pass


if __name__ == '__main__':
    data, data_to_train, data_to_test = split_data(
            ticker, start_date, end_date, features_list
    )

    run_LSTM_based_model(data, data_to_train, data_to_test, name_of_sector, ticker, dropout, epochs, batch_size)
    
