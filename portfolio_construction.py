import os, re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib, yaml
import warnings

from Pipeline.get_data import *

warnings.simplefilter(action='ignore', category=Warning)

DATETIME_RUN = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# plotting parameters
plt.rcParams['figure.figsize'] = (16,9)
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 12})


start_date = datetime.datetime(2016, 1, 1)
end_date = datetime.datetime(2023, 11, 30)
data_for_index = get_data_by_ticker('BDX.WA', start_date, end_date, ['Close']).index

# name_of_method = 'LSTM'
# name_of_submethod = 'BatchNormalization'
name_of_method = 'ARIMA'
name_of_submethod = 'walk_forward_validation'


# structure of folders
# \Results\wig-budownictwo\BDX.WA\LSTM\BatchNormalization\2023-12-31_01-05-32
def _max_datetime_folder(name_of_sector, ticker):
    """ Returns name of folder with results that date of its creation is the latest """
    rootdir = os.path.join('Results', name_of_sector, ticker, name_of_method, name_of_submethod)

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


def calculate_real_and_predicted_by_LSTM_returns(sector_pred_returns_for_portfel, sector_returns_for_portfel, data, y_test_predictions, ticker):
    # Zwroty używając predykcji - cena dzisiejsza znana, na jutro predykcja
    data_test_values = data[-(len(y_test_predictions)+1):].flatten()
    # per_growth = [(((y - x) * 100) / x) for x, y in zip(data_test_values, y_test_predictions)]
    per_growth = (y_test_predictions.flatten() - data_test_values[:-1]) / data_test_values[:-1] * 100

    sector_pred_returns_for_portfel[ticker+'_returns'] = per_growth.flatten()

    # Wariancja na cenach rzeczywistych - przygotowanie danych
    prices_for_var = data[-(len(data_test_values) + 50):].flatten()
    data_for_var = pd.DataFrame(prices_for_var, columns = ['data'])
    data_for_var = data_for_var['data'].pct_change().dropna() * 100
    data_for_var = data_for_var.reset_index(drop=True)

    # vars = [np.var(data_for_var[i:i+50]) for i in range(len(data_for_var)-50)]
    sector_returns_for_portfel[ticker+'_real_data_ret'] = data_for_var

    return sector_pred_returns_for_portfel, sector_returns_for_portfel


def _save_portfolio_plot_and_weights(results_frame, max_sharpe_port, min_vol_port, predicted_portfolio, day_num):
    if predicted_portfolio:
        model_save_dir_plots = os.path.join("Results", "Portfolios", DATETIME_RUN, name_of_sector, name_of_method, "Plots")
        model_save_dir_data = os.path.join("Results", "Portfolios", DATETIME_RUN, name_of_sector, name_of_method, "Data")
    else:
        model_save_dir_plots = os.path.join("Results", "Portfolios", DATETIME_RUN, name_of_sector, "Real_Data", "Plots")
        model_save_dir_data = os.path.join("Results", "Portfolios", DATETIME_RUN, name_of_sector, "Real_Data", "Data")

    os.makedirs(model_save_dir_plots, exist_ok=True)
    os.makedirs(model_save_dir_data, exist_ok=True)

    results_frame.to_csv(os.path.join(model_save_dir_data, "portfolio_" + name_of_sector + name_of_method + str(day_num) + ".csv"))
    
    #create scatter plot coloured by Sharpe Ratio
    plt.scatter(results_frame.stdev,results_frame.ret,c=results_frame.sharpe,cmap='RdYlBu')
    plt.xlabel('Ryzyko')
    plt.ylabel('Oczekiwany procentowy zwrot')
    plt.colorbar()
    #plot red star to highlight position of portfolio with highest Sharpe Ratio
    plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=500, label="Portfel optymalny")
    #plot green star to highlight position of minimum variance portfolio
    plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=500, label="Portfel o minimalnym ryzyku")
    plt.legend()
    if predicted_portfolio:
        plt.title(f"Warianty portfeli dla sektora {name_of_sector} skonstruowane przy użyciu predykcji z modelu {name_of_method}", fontsize = 15)
    else:
        plt.title(f"Warianty portfeli dla sektora {name_of_sector} skonstruowane przy użyciu danych rzeczywistych)", fontsize = 15)
    plt.savefig(os.path.join(model_save_dir_plots, "portfolio_" + name_of_sector + name_of_method + str(day_num) + ".png"))
    plt.clf()


def construct_daily_portfolio(sector_pred_returns_for_portfel, sector_returns_for_portfel, day, predicted_portfolio=True):
    column_names = list(sector_pred_returns_for_portfel.filter(regex=".*_returns.*").columns.values)
    slice_object = slice(3)
    ticker_list = [i[slice_object] for i in column_names]

    num_portfolios = 10000
    results = np.zeros((len(ticker_list) + 3, num_portfolios))
    for i in range(num_portfolios):
        #select random weights for portfolio holdings
        weights = np.array(np.random.random(5))
        #rebalance weights to sum to 1
        weights /= np.sum(weights)

        #calculate portfolio return and volatility
        real_returns = sector_returns_for_portfel.filter(regex=".*_real_data_ret.*")[day:50+day]
        cov_matrix = real_returns.cov()
        if predicted_portfolio:
            stocks_returns = sector_pred_returns_for_portfel.filter(regex=".*_returns.*").iloc[day, :]
        else:
            stocks_returns = sector_returns_for_portfel.filter(regex=".*_real_data_ret.*").iloc[50+day, :]
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

    #convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T,columns=['ret','stdev','sharpe',ticker_list[0],ticker_list[1],ticker_list[2],ticker_list[3], ticker_list[4]])
    #locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    #locate positon of portfolio with minimum standard deviation
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    
    _save_portfolio_plot_and_weights(results_frame, max_sharpe_port, min_vol_port, predicted_portfolio, day)

    # print(max_sharpe_port)


if __name__ == '__main__':

    with open('config.yaml') as f:
        ticker_dict = yaml.safe_load(f)

    for name_of_sector in ticker_dict:
        try:
            print("Starting for sector:", name_of_sector)
            sector_returns_for_portfel = pd.DataFrame()
            sector_pred_returns_for_portfel = pd.DataFrame()
            for ticker in ticker_dict[name_of_sector]:
                target_datetime_dir = _max_datetime_folder(name_of_sector, ticker)
                target_dir_path = os.path.join('Results', name_of_sector, ticker, name_of_method, name_of_submethod, target_datetime_dir, 'Data')

                for root, dirs, files in os.walk(target_dir_path, topdown=False):
                    if name_of_method == 'LSTM':
                        _data_results_dict = load_data_results(files)
                        data, x_test, x_train, y_test, y_test_predictions, y_train, y_train_predictions = [
                            _data_results_dict[name]
                            for name in
                            ["data.npy", "x_test.npy", "x_train.npy", "y_test.npy", "y_test_predictions.npy", "y_train.npy", "y_train_predictions.npy"]
                        ]

                    else:
                        _data_results_dict = load_data_results(files)
                        data_to_test, data_to_train, params, predictions = [
                            _data_results_dict[name]
                            for name in
                            ["data_to_test.npy", "data_to_train.npy", "params.npy", "predictions.npy"]
                        ]
                        y_test_predictions = predictions
                        data = get_data_by_ticker(ticker, start_date, end_date, ['Close']).values
                        while data.shape == (0,1):
                            print(f"Retrying to get data for ticker {ticker}")
                            data = get_data_by_ticker(ticker, start_date, end_date, ['Close']).values 

                sector_pred_returns_for_portfel, sector_returns_for_portfel = calculate_real_and_predicted_by_LSTM_returns(sector_pred_returns_for_portfel, 
                                                                                                                        sector_returns_for_portfel, data, y_test_predictions, ticker)

            for day in range(len(sector_pred_returns_for_portfel)):
                if day % 100 == 0:
                    print(f"Day {day} of {len(sector_pred_returns_for_portfel)} for sector {name_of_sector}")
                construct_daily_portfolio(sector_pred_returns_for_portfel, sector_returns_for_portfel, day, predicted_portfolio=True)
                construct_daily_portfolio(sector_pred_returns_for_portfel, sector_returns_for_portfel, day, predicted_portfolio=False)
        except:
            pass


                



