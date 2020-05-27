# imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
# Classes
from CNN import CNN
from LSTM import LSTM_
from data import Data

def main():
    #Load the dataset
    dataset = Data(path='dataset/Train.csv')
    temp = dataset.getDataframe()
    # TODO remove > 20% nans
    # TODO avergae the rest of the nans
    # TODO Split into testing a training sets.
    # TODO create average and std function
    # TODO create loops for splitting, training and testing
    #CNN labels, training and testing
    # arrLabelsCNN = np.array([[float(i)] for i in temp['target'].values])
    # arrDataCNN = dataset.convertToNpArrCNN()
    # cnn = CNN(inputShape=(6,121,1))
    # cnn.train(trainInputs=arrData,trainOutputs=arrLabels,epoches=300,valSplit=0.15,batchSize=50)

    #LSTM labels, training and testing
    arrDataLSTM = dataset.convertToNpArrLSTM()
    arrLabelsLSTM = np.array([[float(i)] for i in temp['target'].values])
    lstm = LSTM_(inputShape=(121,6))
    lstm.train(trainInputs=arrDataLSTM,trainOutputs=arrLabelsLSTM, epoches=300, valSplit=0.15, batchSize=50)

    return


if __name__ == "__main__":
    main()
