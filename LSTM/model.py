from keras.models import Sequential, load_model
import tensorflow as tf
from keras.layers import LSTM, Dense, Flatten, BatchNormalization
import keras.metrics

# num_companies = 1
# lookback = 50 #How many days of the past can the model see
# horizon = 1 #How many days into the future are we trying to predict

#Model Architecture - Two LSTM layers with x neurons & y epochs
num_neurons_L1 = 256
num_neurons_L2 = 256
num_neurons_dense1 = 1
num_features = 1


def build_model(num_features, lookback, horizon, learning_rate, dropout):
    model = Sequential()

    model.add(LSTM(units = num_neurons_L1, input_shape=(lookback,num_features), return_sequences=True, activation = 'relu', dropout=dropout))
    model.add(LSTM(num_neurons_L2, activation = 'relu', dropout=dropout, return_sequences=False))
    model.add(Flatten())
    model.add(Dense(horizon, activation = 'sigmoid'))
    model.add(BatchNormalization())

    model.compile(loss='Huber', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics = ['mean_absolute_error',
                                                                                                            'mse',
                                                                                                            keras.metrics.RootMeanSquaredError(),
                                                                                                            keras.metrics.MeanAbsolutePercentageError(),
                                                                                                            keras.metrics.MeanSquaredLogarithmicError(),
                                                                                                            keras.metrics.LogCoshError()
                                                                                                            ])

    return model

