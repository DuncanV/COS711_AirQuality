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

    def convertToNpArrCNN(self, data):
        # creates array of (?, 6, 121, 1)
        entireArr = []
        rowArr = []
        item = []
        for row in data.values:
            rowArr = []
            for attribute in row[2:8]:
                rowArr.append([[float(i)] for i in attribute.replace("nan", "-100").split(',')])
            entireArr.append(rowArr.copy())
        CNNarray = np.array(entireArr)
        return CNNarray

    def convertToNpArrLSTM(self, data):
        # creates array of (?, 121, 6)
        entireArr = []
        rowArr = []
        item = []
        for row in data.values:
            rowArr = []
            for attribute in row[2:8]:
                rowArr.append([float(i) for i in attribute.replace("nan", "-100").split(',')])
            entireArr.append(rowArr.copy())
        LSTMarray = np.array(entireArr).transpose(0, 2, 1)
        return LSTMarray

    def getLabels(self, target):
        pass

    def getCNNDataArr(self):
        return self.CNNarray

    def getLSTMDataArr(self):
        return self.LSTMarray

    def removeNans(self, AMOUNT=0.2):
        pass
