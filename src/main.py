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
    SPLIT = 0.2
    LOOP = 10
    # Load the dataset
    dataset = Data(path='dataset/Train.csv')
    dataset.removeNans()
    # TODO remove > 20% nans
    # TODO avergae the rest of the nans

    for i in range(LOOP):
        trainDataframe, testDataframe = model_selection.train_test_split(dataframe, test_size=0.2)

        # CNN labels, training and testing
        # arrLabelsCNN = np.array([[float(i)] for i in temp['target'].values])
        # arrDataCNN = dataset.convertToNpArrCNN()
        # cnn = CNN(inputShape=(6,121,1))
        # cnn.train(trainInputs=arrData,trainOutputs=arrLabels,epoches=300,valSplit=0.15,batchSize=50)

        # LSTM labels, training and testing
        # arrDataLSTM = dataset.convertToNpArrLSTM()
        # arrLabelsLSTM = np.array([[float(i)] for i in temp['target'].values])
        # lstm = LSTM_(inputShape=(121,6))
        # lstm.train(trainInputs=arrDataLSTM,trainOutputs=arrLabelsLSTM, epoches=300, valSplit=0.15, batchSize=50)

    return


if __name__ == "__main__":
    main()
