# imports
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from sklearn import model_selection
# Classes
from CNN import CNN
from LSTM import LSTM_
from data import Data


def main():
    TESTSPLIT = 0.2
    VALSPLIT = 0.15
    LOOP = 10
    # Load the dataset
    dataset = Data(path='dataset/Train.csv')
    dataset.removeNans(AMOUNT=0.2, recreate=False)
    dataset.standardize(recreate=False)

    for i in range(LOOP):
        trainDataframe, testDataframe = model_selection.train_test_split(dataset.getDataframe(), test_size=TESTSPLIT)
        trainDataframe, valDataframe = model_selection.train_test_split(trainDataframe, test_size=VALSPLIT)
        # CNN labels, training and testing
        trainLabelsCNN = np.array([[float(i)] for i in trainDataframe['target'].values])
        trainDataCNN = dataset.convertToNpArrCNN(trainDataframe)

        valLabelsCNN = np.array([[float(i)] for i in valDataframe['target'].values])
        valDataCNN = dataset.convertToNpArrCNN(valDataframe)

        testLabelsCNN = np.array([[float(i)] for i in testDataframe['target'].values])
        testDataCNN = dataset.convertToNpArrCNN(testDataframe)
        cnn = CNN(inputShape=(6,121,1))

        cnn.train(trainInputs=trainDataCNN,trainOutputs=trainLabelsCNN,valInputs=valDataCNN, valOutputs=valLabelsCNN
                  ,epoches=300,batchSize=50)

        CNNarray.append(cnn.test(data=testDataCNN, labels=testLabelsCNN))

        # LSTM labels, training and testing
        # arrDataLSTM = dataset.convertToNpArrLSTM()
        # arrLabelsLSTM = np.array([[float(i)] for i in temp['target'].values])
        # lstm = LSTM_(inputShape=(121,6))
        # lstm.train(trainInputs=arrDataLSTM,trainOutputs=arrLabelsLSTM, epoches=300, valSplit=0.15, batchSize=50)

    return


if __name__ == "__main__":
    main()
