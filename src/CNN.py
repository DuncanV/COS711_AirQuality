# imports
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, ZeroPadding2D, Input, Dropout

# inherit
from Network import Network


class CNN(Network):
    def __init__(self, inputShape):
        self.activation_func = 'relu'
        model = Sequential()
        # pad so the kernal sizes will work
        model.add(ZeroPadding2D(padding=(5, 5), input_shape=inputShape))

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation=self.activation_func))
        model.add(MaxPool2D(strides=2))

        model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation=self.activation_func))
        model.add(MaxPool2D(strides=2))

        model.add(Flatten())

        model.add(Dense(120, activation=self.activation_func))
        model.add(Dropout(0.2))

        model.add(Dense(84, activation=self.activation_func))
        model.add(Dropout(0.2))

        model.add(Dense(1, activation='relu'))

        super().__init__(_model=model, _name="LeNet-5")
        return
