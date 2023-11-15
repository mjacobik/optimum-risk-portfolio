from keras.models import Sequential, load_model
import tensorflow as tf
from keras.layers import LSTM,Dense,Flatten, Activation,Attention,Dropout
from keras import optimizers
import os

# num_companies = 1
# lookback = 50 #How many days of the past can the model see
# horizon = 1 #How many days into the future are we trying to predict

#Model Architecture - Two LSTM layers with x neurons & y epochs
num_neurons_L1 = 256
num_neurons_L2 = 256
num_neurons_dense1 = 1


def build_model(num_features, lookback, horizon, learning_rate):
    model = Sequential()

    model.add(LSTM(units = num_neurons_L1, input_shape=(lookback,num_features), return_sequences=True, activation = 'relu', dropout=0.3))
    model.add(LSTM(num_neurons_L2, activation = 'relu', dropout=0.3, return_sequences=False))
    # model.add(Flatten())
    model.add(Dense(horizon, activation = 'sigmoid'))

    model.compile(loss='Huber', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics = ['mean_absolute_error'])

    return model


def save_model_to_json(model, path):
    model_as_json = model.to_json()

    return model_as_json


def save_weights(model, path):
    pass