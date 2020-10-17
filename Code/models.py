import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import pandas as pd
import numpy as np
import math

from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def getPerceptronRegression(layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(neurons,), activation='sigmoid', use_bias=False))

    for i in range(layers):
        model.add(Dense(10, activation='sigmoid', use_bias=False))
        model.add(Dropout(0.1))

    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error');
    return model


def getPerceptronClassification(layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(neurons,), activation='linear', use_bias=False))

    for i in range(layers):
        model.add(Dense(10, activation='relu', use_bias=False))

    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']);
    return model

def getLSTM(look_back):
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
