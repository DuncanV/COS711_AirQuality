# imports
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
# inherit
from Network import Network

class LSTM(Network):
    def __init__(self, inputShape):
        regressor = Sequential()

        regressor.add(LSTM(units=50, return_sequences=True, input_shape=inputShape))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units=50, activation='relu'))
        regressor.add(Dropout(0.2))

        regressor.add(Dense(units=1, activation='relu'))
        super().__init__(_model=[], _name="LSTM")
        return
