# imports
import pandas as pd
import numpy as np
import os

class Data:

    def __init__(self, path=''):
        if path != '':
            self.setDataPath(path)
        else:
            self.data = None
        self.CNNarray = None
        self.LSTMarray = None
        return

    def setDataPath(self, path):
        if not os.path.exists(path):
            print("File does not exits!")
            return
        self.data = pd.read_csv(path)
        return self.data

    def setDatasetframe(self, dataset):
        self.data = dataset
        return self.data

    def getDataframe(self):
        return self.data

    def show(self):
        return print(self.data)

    def convertToNpArrCNN(self):
        print(self.data)
        entireArr = []
        rowArr = []
        item = []
        for row in self.data.values:
            rowArr = []
            for attribute in row[2:8]:
                rowArr.append([[float(i)] for i in attribute.replace("nan", "-100").split(',')])
            entireArr.append(rowArr.copy())
        self.CNNarray = np.array(entireArr)
        return self.CNNarray

    def convertToNpArrLSTM(self):
        # TODO need to create array of (15539, 121, 6)
        pass

    def getLabels(self, target):
        pass

    def getCNNDataArr(self):
        return self.CNNarray

    def getLSTMDataArr(self):
        return self.LSTMarray



