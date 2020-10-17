import  pandas as pd
import numpy as np
import math

def readFile(path):
    input = pd.read_csv(f"data/{path}");

    if (path == "WIG20.csv"):
        return (input["Zamkniecie"], input["Data"]);


def normalizeToRatios(df):
    res = df.diff() / df[1:];
    res = res[1: len(df)]
    return res


def normalizeToFirst(df):
    first = df[0]
    res = df / first
    return res


# Create input dataset in form:
# x_1 x_2 x_3    x_4
# x_2 x_3 x_4    x_5
# x_3 x_4 x_5    x_6
# ..................
def create_dataset(dataset, look_back=3, look_in_future=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_in_future):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back + look_in_future])

    return (np.array(dataX), np.array(dataY))


# Create training, validation and test sets
# Based on create_dataset function
def createSets(df, look_back = 10, look_in_future = 1):
    X, Y = create_dataset(df, look_back, look_in_future)

    setLen = len(Y)
    trainLen = math.floor(0.7 * setLen)
    validationLen = math.floor(0.1 * setLen)

    trainX = X[0: trainLen]
    trainY = Y[0: trainLen]
    validationX = X[trainLen: trainLen + validationLen]
    validationY = Y[trainLen: trainLen + validationLen]
    testX = X[trainLen + validationLen: setLen]
    testY = Y[trainLen + validationLen: setLen]

    return (trainX, trainY, validationX, validationY, testX, testY)

def flatten(l):
    return [item for sublist in l for item in sublist]
