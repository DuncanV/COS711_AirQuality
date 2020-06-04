# imports
import os
import math
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import  optimizers
import sklearn as sk
import statistics


class Network:

    def __init__(self, _model, _name):
        self.model = _model
        self.name = _name
        self.compiled = False
        return

    def compile(self):
        lr = 0.001
        optimizer = optimizers.Adam(learning_rate=lr)
        # optimizer = optimizers.Adadelta(learning_rate=lr)
        # optimizer = optimizers.Adamax(learning_rate=lr)

        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae'])
        return

    def train(self, trainInputs, trainOutputs, valInputs, valOutputs, epoches, batchSize):
        print('[INFO] Training ' + self.name + ' on dataset')
        if not self.compiled:
            self.compile()
            earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=epoches // 10,
                                      verbose=0, mode='auto')
            bestModel = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',
                                        save_best_only=True)
        history = self.model.fit(trainInputs, trainOutputs, epochs=epoches, validation_data=(valInputs, valOutputs) , verbose=1,
                                 batch_size=batchSize,
                                 callbacks=[earlyStop, bestModel])
        self.model.load_weights('best_model.h5')
        return

    def test(self, data, labels):
        print('[INFO] Testing ' + self.name + ' on dataset')
        testpred = self.model.predict(data)
        mse = sk.metrics.mean_squared_error(labels.flatten(), testpred.flatten())
        rmse = math.sqrt(mse)
        return rmse

    def getModel(self):
        if not self.compiled:
            self.compile()
        return self.model

    def saveWeights(self, dataset):
        if os.path.exists("weights/" + dataset + self.name + ".h5"):
            os.remove("weights/" + dataset + self.name + ".h5")
        self.model.save_weights("weights/" + dataset + self.name + ".h5")
        print("[INFO] Weights saved for " + self.name + " on " + dataset + " Dataset.")
        return

    def loadWeights(self, dataset):
        if not os.path.exists("weights/" + dataset + self.name + ".h5"):
            print("[ERROR] Weights not found for " + self.name + " on " + dataset + " Dataset.")
            return False
        else:
            self.model.load_weights("weights/" + dataset + self.name + ".h5")
            print("[INFO] Weights loaded for " + self.name + " on " + dataset + " Dataset.")
            return True

    def avgStd(self, arr):
        print('[RESULTS] Results for ' + self.name)
        print("Average: ", round(statistics.mean(arr), 6))
        print("Standard Dev: ", round(statistics.stdev(arr), 6))
        return