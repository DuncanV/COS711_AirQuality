#imports
import os


class Network:

    def __int__(self, _model, _name):
        print("In Network")
        self.model = _model
        self.name = _name
        self.compiled = False
        return

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return

    def train(self):
        pass

    def test(self):
        pass

    def getModel(self):
        if not self.compiled:
            self.compile()
        return self.model

    def saveWeights(self, dataset):
        if os.path.exists("weights/" + dataset + self.name + ".h5"):
            os.remove("weights/" + dataset + self.name + ".h5")
        self.model.save_weights("weights/" + dataset + self.name + ".h5")
        print("Weights saved for " + self.name + " on " + dataset + " Dataset.")
        return

    def loadWeights(self, dataset):
        if not os.path.exists("weights/" + dataset + self.name + ".h5"):
            print("Weights not found for " + self.name + " on " + dataset + " Dataset.")
            return False
        else:
            self.model.load_weights("weights/" + dataset + self.name + ".h5")
            print("Weights loaded for " + self.name + " on " + dataset + " Dataset.")
            return True
