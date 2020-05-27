# imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
# Classes
from CNN import CNN
from LSTM import LSTM
from data import Data

def main():
    #Load the dataset
    dataset = Data(path='dataset/Train.csv')
    testData = Data(path='dataset/Test.csv')

    attributes = ['temp','precip','rel_humidity','wind_dir','wind_spd','atmos_press']
    temp = dataset.getDataframe()
    # arrLabels = arrLabels['target'].values.astype(np.float32)
    arrLabels = np.array([[float(i)] for i in temp['target'].values])
    arrData = dataset.convertToNpArrCNN()
    print(arrData.shape)
    # print(arrLabels.shape)
    print(arrLabels)
    # print("==========================================")
    # arrData = dataset.standardise()
    # print(arrData.shape)

    # cnn = CNN(inputShape=(6,121,1))
    # cnn.train(trainInputs=arrData,trainOutputs=arrLabels,epoches=300,valSplit=0.15,batchSize=50)


    return


if __name__ == "__main__":
    main()
