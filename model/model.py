from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D


class CNN:
    def __init__(self, config):
        self.config = config
    
    def get_model(self):
        model = Sequential([
            Conv2D(32, kernel_size=(2, 2), input_shape=self.config.shape),
            Conv2D(32, kernel_size=(3, 1)),
            Flatten(),
            Dense(32),
            Dense(8),
            Activation('softmax'),
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model

