import numpy as np
import math

def loadData(filePath, seed=None, S=None):
    if seed is None:
        np.random.seed(0)
    else:
        np.random.seed(seed)
    data = np.genfromtxt(filePath, dtype=None, delimiter=',', encoding=None, names=True)
    np.random.shuffle(data)
    if S is None:
        seperate = np.array_split(data, 3)
        return np.concatenate((seperate[0], seperate[1])), seperate[2]
    else:
        seperate = np.array_split(data, S)
        return seperate

def enumerateData(data):
    _, enumerated = np.unique(data, return_inverse=True)
    return enumerated

def preprocessData(data):
    newData = np.array([
        np.ones_like(data["age"]),
        data["age"],
        enumerateData(data["sex"]),
        data["bmi"],
        data["children"],
        enumerateData(data["smoker"]),
        enumerateData(data["region"]),
        data["charges"]
    ]).transpose()

    return newData

def closedFormLinearRegression(trainX, trainY, validationX, validationY, isTrain): # always train on x, change the yhat's x based on what you wanna calculate
    w = np.dot(np.linalg.inv(np.dot(trainX.transpose(), trainX)), np.dot(trainX.transpose(), trainY))

    if isTrain:
        N = len(trainY)
        yHat = np.dot(trainX, w)

        rmse = math.sqrt((1 / N) * np.dot(np.subtract(trainY, yHat).transpose(), np.subtract(trainY, yHat)))
        smape = (1 / N) * np.sum(abs(np.subtract(trainY, yHat)) / (abs(trainY) + abs(yHat)))
        se = (1 / N) * np.dot(np.subtract(trainY, yHat).transpose(), np.subtract(trainY, yHat))
    else:
        N = len(validationY)
        yHat = np.dot(validationX, w)

        rmse = math.sqrt((1 / N) * np.dot(np.subtract(validationY, yHat).transpose(), np.subtract(validationY, yHat)))
        smape = (1 / N) * np.sum(abs(np.subtract(validationY, yHat)) / (abs(validationY) + abs(yHat)))
        se = (1 / N) * np.dot(np.subtract(validationY, yHat).transpose(), np.subtract(validationY, yHat))
        
    return rmse, smape, se

def combineFolds(folds):
    return np.concatenate(folds)

def getXY(data):
    X = data[:len(data), :len(data[0])-1]
    Y = data[:len(data), len(data[0])-1]
    return X, Y

def crossValidationLinearRegression(dataPath, S):
    squaredErrors = []
    rmse = []
    for currentSeed in range(1, 21):
        data = loadData(dataPath, seed=currentSeed, S=S)
        for i in range(0, S):
            validationData = preprocessData(data[i])
            trainData = preprocessData(combineFolds(np.delete(data, i, axis=0)))
            Xtrain, Ytrain = getXY(trainData)
            Xvalidation, Yvalidation = getXY(validationData)
            _, _, se = closedFormLinearRegression(Xtrain, Ytrain, Xvalidation, Yvalidation, False) # false uses validation for yhat and calculations
            squaredErrors.append(se)
        rmse.append(math.sqrt(np.mean(squaredErrors, axis=0)))
        squaredErrors = []
    return np.mean(rmse, axis=0), np.std(rmse, axis=0, ddof=1)

def main():
    dataPath = "insurance.csv"

    train, validation = loadData(dataPath)
    trainData = preprocessData(train)
    validationData = preprocessData(validation)

    Xtrain, Ytrain = getXY(trainData)
    Xvalidation, Yvalidation = getXY(validationData)

    # first two paramaters are used for creating the learning model w 
    # second two are used to predict and calculate RMSE, SMAPE, SE
    trainRmse, trainSmape, _ = closedFormLinearRegression(Xtrain, Ytrain, Xvalidation, Yvalidation, True)
    validationRmse, validationSmape, _ = closedFormLinearRegression(Xtrain, Ytrain, Xvalidation, Yvalidation, False)

    print("TRAINING DATA\n RMSE:", trainRmse, "\n", "SMAPE:", trainSmape)
    print("VALIDATION DATA\n RMSE:", validationRmse, "\n", "SMAPE:", validationSmape)

    mean, std = crossValidationLinearRegression(dataPath, S=223)
    print("Mean:", mean)
    print("Standard Deviation:", std)

if __name__ == "__main__":
    main()