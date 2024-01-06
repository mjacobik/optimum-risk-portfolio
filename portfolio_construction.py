import os, re
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

name_of_method = 'LSTM'
name_of_submethod = 'BatchNormalization'


# structure of folders
# \Results\wig-budownictwo\BDX.WA\LSTM\BatchNormalization\2023-12-31_01-05-32
def _max_datetime_folder(name_of_sector, ticker):
    """ Returns name of folder with results that date of its creation is the latest """
    rootdir = f'Results\{name_of_sector}\{ticker}\{name_of_method}\{name_of_submethod}'

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


def _save_portfolio_plot(results_frame, max_sharpe_port, min_vol_port):
    model_save_dir = os.path.join("Results", 
                                  name_of_sector)

    print(model_save_dir)
    #create scatter plot coloured by Sharpe Ratio
    plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
    plt.xlabel('Ryzyko')
    plt.ylabel('Oczekiwany zwrot')
    plt.colorbar()
    #plot red star to highlight position of portfolio with highest Sharpe Ratio
    plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=700)
    #plot green star to highlight position of minimum variance portfolio
    plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=700)
    plt.title(f"Portfel inwestycyjny dla sektora {name_of_sector} skonstruowany przy użyciu modelu {name_of_method}", fontsize = 15)
    plt.savefig(os.path.join(model_save_dir, "portfolio_" + name_of_sector + name_of_method + ".png"))
    plt.clf()


def construct_daily_portfolio(sector_data_for_portfel, predicted_portfolio=True):
    num_portfolios = 10000
    results = np.zeros((4+5-1,num_portfolios))
    for i in range(num_portfolios):
        #select random weights for portfolio holdings
        weights = np.array(np.random.random(5))
        #rebalance weights to sum to 1
        weights /= np.sum(weights)

        #calculate portfolio return and volatility
        real_returns = sector_data_for_portfel.filter(regex=".*_real_data_ret.*")[0:50]
        cov_matrix = real_returns.cov()
        if predicted_portfolio:
            stocks_returns = sector_data_for_portfel.filter(regex=".*_returns.*").iloc[0, :]
        else:
            stocks_returns = sector_data_for_portfel.filter(regex=".*_real_data_ret.*").iloc[0, :]
        portfolio_return = np.sum(stocks_returns * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights)))

        #store results in results array
        results[0,i] = portfolio_return
        results[1,i] = portfolio_std_dev
        #store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2,i] = (results[0,i] - 0.01) / results[1,i]

        #iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results[j+3,i] = weights[j]


    column_names = list(sector_data_for_portfel.filter(regex=".*_returns.*").columns.values)
    slice_object = slice(3)
    ticker_list = [i[slice_object] for i in column_names]

    #convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',ticker_list[0],ticker_list[1],ticker_list[2],ticker_list[3], ticker_list[4]])
    #locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    #locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    
    _save_portfolio_plot(results_frame, max_sharpe_port, min_vol_port)

    print(max_sharpe_port)
    return max_sharpe_port


if __name__ == '__main__':

    with open('config.yaml') as f:
        ticker_dict = yaml.safe_load(f)

    for name_of_sector in ticker_dict:
        sector_data_for_portfel = pd.DataFrame()
        for ticker in ticker_dict[name_of_sector]:
            target_datetime_dir = _max_datetime_folder(name_of_sector, ticker)
            target_dir_path = f'Results\{name_of_sector}\{ticker}\{name_of_method}\{name_of_submethod}\{target_datetime_dir}\Data'

            for root, dirs, files in os.walk(target_dir_path, topdown=False):
                if name_of_method == 'LSTM':
                    data, x_test, x_train, y_test, y_test_predictions, y_train, y_train_predictions = load_data_results(files).values()
                else:
                    data_to_test, data_to_train, params, predictions = load_data_results(files).values()
                    data = data_to_test
                    y_test_predictions = predictions        

                # Zwroty używając predykcji - cena dzisiejsza znana, na jutro predykcja
                data_test_values = data[-(len(y_test_predictions)+1):]
                # per_growth = [(((y - x) * 100) / x) for x, y in zip(data_test_values, y_test_predictions)]
                per_growth = (y_test_predictions - data_test_values[:-1]) / data_test_values[:-1]
                sector_data_for_portfel[ticker+'_returns'] = per_growth.flatten()

                # Wariancja na cenach rzeczywistych - przygotowanie danych
                prices_for_var = data[-(len(data_test_values) + 50):].flatten()
                data_for_var = pd.DataFrame(prices_for_var, columns = ['data'])
                data_for_var = data_for_var['data'].pct_change().dropna()
                data_for_var = data_for_var.reset_index(drop=True)

                # vars = [np.var(data_for_var[i:i+50]) for i in range(len(data_for_var)-50)]
                sector_data_for_portfel[ticker+'_real_data_ret'] = data_for_var
        
        construct_daily_portfolio(sector_data_for_portfel)

                



