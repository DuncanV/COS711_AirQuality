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
    LOOP = 20
    # Load the dataset
    dataset = Data(path='dataset/Train.csv')
    dataset.removeNans(AMOUNT=0.2, recreate=False)
    dataset.standardize(recreate=False)
    trainarrCNN = []
    valarrCNN = []
    testarrCNN = []
    testarrLSTM = []
    trainarrLSTM = []
    valarrLSTM = []
    cnn = None
    print("[INFO] Starting Loop...")
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
        cnn = CNN(inputShape=(6, 121, 1))

        cnn.train(trainInputs=trainDataCNN, trainOutputs=trainLabelsCNN, valInputs=valDataCNN, valOutputs=valLabelsCNN
                  , epoches=300, batchSize=50)

        testarrCNN.append(cnn.test(data=testDataCNN, labels=testLabelsCNN))
        trainarrCNN.append(cnn.test(data=trainDataCNN, labels=trainLabelsCNN))
        valarrCNN.append(cnn.test(data=valDataCNN, labels=valLabelsCNN))

        # LSTM labels, training and testing
        trainLabelsLSTM = np.array([[float(i)] for i in trainDataframe['target'].values])
        trainDataLSTM = dataset.convertToNpArrLSTM(trainDataframe)

        valLabelsLSTM = np.array([[float(i)] for i in valDataframe['target'].values])
        valDataLSTM = dataset.convertToNpArrLSTM(valDataframe)

        testLabelsLSTM = np.array([[float(i)] for i in testDataframe['target'].values])
        testDataLSTM = dataset.convertToNpArrLSTM(testDataframe)
        lstm = LSTM_(inputShape=(121, 6))
        lstm.train(trainInputs=trainDataLSTM, trainOutputs=trainLabelsLSTM, valInputs=valDataLSTM,
                   valOutputs=valLabelsLSTM, epoches=300, batchSize=50)

        testarrLSTM.append(lstm.test(data=testDataLSTM, labels=testLabelsLSTM))
        trainarrLSTM.append(lstm.test(data=trainDataLSTM, labels=trainLabelsLSTM))
        valarrLSTM.append(lstm.test(data=valDataLSTM, labels=valLabelsLSTM))

    print()
    cnn.avgStd(trainarrCNN, 'Training CNN')
    cnn.avgStd(valarrCNN, 'Val CNN')
    cnn.avgStd(testarrCNN, 'Testing CNN')
    cnn.avgStd(trainarrLSTM, 'Training LSTM')
    cnn.avgStd(valarrLSTM, 'Val LSTM')
    cnn.avgStd(testarrLSTM, 'Testing LSTM')
    return


if __name__ == "__main__":
    main()
