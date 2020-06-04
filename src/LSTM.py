# imports
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
# inherit
from Network import Network


class LSTM_(Network):
    def __init__(self, inputShape):
        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=inputShape))
        model.add(Dropout(0.2))

        model.add(LSTM(units=25, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=10))
        model.add(Dropout(0.2))

        model.add(Dense(units=1, activation='relu'))

        super().__init__(_model=model, _name="LSTM")
        return
