import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib, yaml

from Pipeline.get_data import *


# plotting parameters
plt.rcParams['figure.figsize'] = (16,9)
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 12})


start_date = datetime.datetime(2016, 1, 1)
end_date = datetime.datetime(2023, 11, 30)
data_for_index = get_data_by_ticker('BDX.WA', start_date, end_date, ['Close']).index

name_of_method = 'ARIMA'
name_of_submethod = 'walk_forward_validation'


# structure of folders
# \Results\wig-budownictwo\BDX.WA\LSTM\BatchNormalization\2023-12-31_01-05-32
def _max_datetime_folder(name_of_sector, ticker):
    """ Returns name of folder with results that date of its creation is the latest """
    rootdir = f'Results\{name_of_sector}\{ticker}_przykład\{name_of_method}\{name_of_submethod}'

    for root, dirs, files in os.walk(rootdir, topdown=False):
        datetime_dirs = []
        for dir in dirs:
            try:
                # datetime_object = datetime.strptime(dir, "%Y-%m-%d_%H-%M-%S")
                datetime_dirs.append(dir)
            except ValueError:
                "Wrong format"

    return max(datetime_dirs)


def load_data_results(files):
    """ Load files Data directory """
    dict = {}
    for name in files:
        base_filename, filename_suffix = os.path.splitext(name)
        if filename_suffix != '.pkl':
            file_content = np.load(open(os.path.join(root, name), 'rb'))
            dict[name] = file_content
        # elif name == 'scaler.pkl':
        #     with open(os.path.join(root, name), 'rb') as f:
        #         dict[name] = joblib.load(f)

    return dict


if __name__ == '__main__':

    with open('config.yaml') as f:
        ticker_dict = yaml.safe_load(f)

    for name_of_sector in ticker_dict:
        for ticker in ticker_dict[name_of_sector]:
            print(ticker)
            target_datetime_dir = _max_datetime_folder(name_of_sector, ticker)
            target_dir_path = f'Results\{name_of_sector}\{ticker}_przykład\{name_of_method}\{name_of_submethod}\{target_datetime_dir}\Data'

            sector_data_for_portfel = []
            for root, dirs, files in os.walk(target_dir_path, topdown=False):
                if name_of_method == 'LSTM':
                    data, x_test, x_train, y_test, y_test_predictions, y_train, y_train_predictions = load_data_results(files).values()
                else:
                    data_to_test, data_to_train, params, predictions = load_data_results(files).values()
                    data = data_to_test
                    y_test_predictions = predictions
                # y_train = scaler.inverse_transform(y_train)
                # y_test = scaler.inverse_transform(y_test)

                model_save_dir = os.path.join("Results", 
                                              name_of_sector,
                                              ticker,
                                              name_of_method,
                                              name_of_submethod)
                
                # returns = pd.DataFrame(y_test).pct_change().dropna()
                # returns['datetime'] = data_for_index[data_for_index.slice_indexer("2023-03-15", "2023-11-30")]


                # 230 to rozmiar zbioru testowego - wszystkich dni kalendarzowych w nim
                arr = np.zeros(230)
                if name_of_method == 'LSTM':
                    arr = np.where(arr==0, np.nan, arr).reshape(-1,1)
                if name_of_method == 'ARIMA':
                    arr = np.where(arr==0, np.nan, arr)
                arr[50:] = y_test_predictions

                if ticker == 'BDX.WA':
                    ticker = 'Budimex S.A.'
                    params = '(5, 2, 0)'
                elif ticker == 'PKO.WA':
                    ticker = 'PKO BP S.A.'
                    params = '(2, 1, 2)'
                
                # LSTM
                # plt.plot(data_for_index[data_for_index.slice_indexer("2023-01-01", "2023-11-30")], data[-230:], label='Dane rzeczywiste testowe')
                plt.plot(data_for_index[data_for_index.slice_indexer("2023-01-01", "2023-11-30")], data, label='Dane rzeczywiste testowe')
                plt.plot(data_for_index[data_for_index.slice_indexer("2023-01-01", "2023-11-30")], arr, label='Predykcja')
                plt.title(f"Predykcja cen akcji spółki {ticker} przy użyciu modelu {name_of_method}{params}", fontsize = 25)
                plt.xlabel("Dzień notowań")
                plt.ylabel("Cena instrumentu")
                plt.legend()
                plt.savefig(os.path.join(model_save_dir, "test_" + ticker + name_of_method + 'recover' + ".png"))
                plt.clf()

