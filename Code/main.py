#%%
from utils import *
from models import *

#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
import math
import pandas as pd

from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

#%%
init_df, date = readFile("WIG20.csv");
date = pd.to_datetime(date)

#%%
#df = df[2000: 2800].reset_index(drop=True)

#%%
#plt.plot(date, init_df)
#plt.show()

#%%
layers = 5;
n_epochs = 100;

def predictWithRegression(df = init_df, look_back = 10, look_in_future = 1):
    tmp_df = df
    df = normalizeToFirst(df)
    trainX, trainY, validationX, validationY, testX, testY = createSets(df, look_back, look_in_future)

    model = getPerceptronRegression(layers, look_back);
    model.fit(trainX, trainY, epochs = n_epochs, validation_data=(validationX, validationY));
    prediction = model.predict(testX)
    prediction = np.array(flatten(prediction))

    # We need to transforms results to see not values of prices, but the movements (whether price has risen or fallen)
    for i in range(len(prediction)):
        if prediction[i] > testX[i][look_back - 1]:
            prediction[i] = 1
        else:
            prediction[i] = -1

    df_ratios = normalizeToRatios(tmp_df)
    trainX_r, trainY_r, validationX_r, validationY_r, testX_r, testY_r = createSets(df_ratios, look_back, look_in_future)

    accuracy = float(sum(prediction * np.array(testY_r) > 0)) / len(prediction)

    return prediction, accuracy



def predictWithClassification(df = init_df, look_back = 10, look_in_future = 1):
    df = normalizeToRatios(df)
    trainX, trainY, validationX, validationY, testX, testY = createSets(df, look_back, look_in_future)

    trainY[trainY >= 0] = 1
    trainY[trainY < 0] = 0
    trainY = keras.utils.to_categorical(trainY)

    validationY[validationY >= 0] = 1
    validationY[validationY < 0] = 0
    validationY = keras.utils.to_categorical(validationY)

    testY[testY >= 0] = 1
    testY[testY < 0] = 0
    testY = keras.utils.to_categorical(testY)

    model = getPerceptronClassification(layers, look_back);
    model.fit(trainX, trainY, epochs = n_epochs, validation_data=(validationX, validationY));

    prediction = np.argmax(model.predict(testX), axis=1)
    testY = np.argmax(testY, axis=1)

    accuracy = float(sum(prediction * np.array(testY) > 0)) / len(prediction)

    return prediction, accuracy

def predictWithLSTM(df = init_df, look_back=10, look_in_future=1):
    df = normalizeToRatios(df)
    trainX, trainY, validationX, validationY, testX, testY = createSets(df, look_back, look_in_future)
    model = getLSTM(look_back);

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    validationX = np.reshape(validationX, (validationX.shape[0], 1, validationX.shape[1]))

    model.fit(trainX, trainY, epochs=n_epochs, validation_data=(validationX, validationY));
    prediction = model.predict(testX)
    prediction = np.array(flatten(prediction))

    accuracy = float(sum(prediction * np.array(testY) > 0)) / len(prediction)

    return prediction, accuracy


# Below part of code was used for experiments. It was changed many times. Currently,
lookBacks = [5, 10, 15, 20, 30, 40, 50]

# We will be working on fragments of input sequence
dflen = len(init_df)
setlen = math.ceil(0.1 * dflen)

accuracies_10 = []
accuracies_20 = []
accuracies_30 = []

for i in range(100):
    randStart = np.random.randint(dflen - setlen)
    pred, acc = predictWithLSTM(init_df[randStart: randStart + setlen].reset_index(drop=True), 10)
    accuracies_10.append(acc)
    accuracies_10.append(np.average(accuracies_10))

    randStart = np.random.randint(dflen - setlen)
    pred, acc = predictWithLSTM(init_df[randStart: randStart + setlen].reset_index(drop=True), 20)
    accuracies_20.append(acc)
    accuracies_20.append(np.average(accuracies_20))

    randStart = np.random.randint(dflen - setlen)
    pred, acc = predictWithLSTM(init_df[randStart: randStart + setlen].reset_index(drop=True), 30)
    accuracies_30.append(acc)
    accuracies_30.append(np.average(accuracies_30))

accur = pd.DataFrame({"accuracies_10": accuracies_10,
                      "accuracies_20": accuracies_20,
                      "accuracies_30": accuracies_30})
