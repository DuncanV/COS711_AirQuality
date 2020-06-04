# imports
from progress.bar import Bar
from statistics import mean, stdev
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
                # replace nan with 0 in case it was missed
                rowArr.append([[float(i)] for i in attribute.replace("nan", "0.0").split(',')])
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
                # replace nan with 0 in case it was missed
                rowArr.append([float(i) for i in attribute.replace("nan", "0.0").split(',')])
            entireArr.append(rowArr.copy())
        LSTMarray = np.array(entireArr).transpose(0, 2, 1)
        return LSTMarray

    def getCNNDataArr(self):
        return self.CNNarray

    def getLSTMDataArr(self):
        return self.LSTMarray

    def removeNans(self, AMOUNT=0.2, recreate=False):
        if os.path.exists('dataset/no_nan.csv') and recreate == False:
            self.data = self.setDataPath('dataset/no_nan.csv')
            return

        # remove nans
        bar = Bar("[INFO] Removing NaN rows...", max=len(self.data.values))
        for row in self.data.values:
            bar.next()
            remove = False
            # count the number of NaNs in the attribute, if > 1-amount
            for attribute in row[2:8]:
                attr = attribute.split(',')
                occurances = attr.count("nan")
                if occurances / len(attr) > (1 - AMOUNT):
                    remove = True
                    break
            # remove the row or replace all nans with average
            if remove == True:
                self.data = self.data[self.data.ID != row[0]]

        # Average the rest
        print()
        bar = Bar("[INFO] Averaging NaN rows...", max=len(self.data.values))
        rows = len(self.data.values)
        for row in range(rows):
            bar.next()
            for attribute in range(2, 8):
                attr = self.data.values[row][attribute].split(',')
                nums = []
                for i in attr:
                    if i != 'nan':
                        nums.append(float(i))

                a = self.data.values[row][attribute].replace("nan", str(mean(nums)))
                self.data.iloc[row, attribute] = a

        print()
        self.data.to_csv('dataset/no_nan.csv', index=False)

    def standardize(self, recreate=False):
        if os.path.exists('dataset/standardise.csv') and recreate == False:
            self.data = self.setDataPath('dataset/standardise.csv')
            return
        columns = list(self.data)
        bar = Bar("[INFO] Standardizing Values...", max=6)
        for col in range(2, 8):
            bar.next()
            allValues = []
            individualValues = []
            for row in range(len(self.data.values)):
                attr = self.data.values[row][col].split(',')
                num = []
                for i in attr:
                    allValues.append(float(i))
                    num.append(float(i))
                individualValues.append(num)
            _mean = mean(allValues)
            _std = stdev(allValues)

            for row in range(len(self.data.values)):
                individualValues[row] = [(x - _mean) / _std for x in individualValues[row]]
                temp = ""
                for item in individualValues[row]:
                    temp += str(item) + ','
                temp = temp[:-1]
                self.data.iloc[row, col] = temp
        print()
        self.data.to_csv('dataset/standardise.csv', index=False)

        return

