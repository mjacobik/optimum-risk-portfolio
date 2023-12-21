import os
from datetime import datetime
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (16,9)
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 12})


def _make_folders_for_results(ticker, name_of_method, name_of_sector):
    model_save_dir = os.path.join(
        "Results",
        name_of_method,
        name_of_sector,
        ticker,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, "Data"), exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, "Model"), exist_ok=True)
    os.makedirs(os.path.join(model_save_dir, "Figures"), exist_ok=True)

    return model_save_dir


def _LSTM_model_training_history_visualise(model_history, model_save_dir):
    """
        Method visualises model's loss function and metrics during training process.
    """
    for _metric in model_history.history.keys():
        _fig, ax = plt.subplots()
        sns.lineplot(x=np.arange(len(model_history.history[_metric]))+1, y=model_history.history[_metric], ax=ax, label=_metric)

        _fig.savefig(os.path.join(model_save_dir, "Figures", _metric + ".png"))
        np.save(os.path.join(model_save_dir, "Model", f"{_metric}.npy"), np.array(model_history.history[_metric]))
        plt.clf()


def save_LSTM_results(ticker, name_of_method, name_of_sector, data, scaler, model, model_history,
                      X_train, Y_train, Y_train_predictions,
                      X_test, Y_test, Y_test_predictions):
    
    model_save_dir = _make_folders_for_results(ticker, name_of_method, name_of_sector)
        
    joblib.dump(scaler, os.path.join(model_save_dir, "Data", "scaler.pkl"))
    _LSTM_model_training_history_visualise(model_history, model_save_dir)
    model.save(os.path.join(model_save_dir, "Model"))

    np.save(os.path.join(model_save_dir, "Data", "data.npy"), np.array(data))

    np.save(os.path.join(model_save_dir, "Data", "x_train.npy"), np.array(X_train))
    np.save(os.path.join(model_save_dir, "Data", "y_train.npy"), np.array(Y_train))
    np.save(os.path.join(model_save_dir, "Data", "y_train_predictions.npy"), np.array(Y_train_predictions))

    np.save(os.path.join(model_save_dir, "Data", "x_test.npy"), np.array(X_test))
    np.save(os.path.join(model_save_dir, "Data", "y_test.npy"), np.array(Y_test))
    np.save(os.path.join(model_save_dir, "Data", "y_test_predictions.npy"), np.array(Y_test_predictions))

    return model_save_dir


def save_weights(model, path):
    pass
