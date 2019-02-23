from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D


class Model:
    def __init__(self, config):
        self.config = config
    
    def get_model(self):
        model = Sequential([
            Conv2D(16, kernel_size=(4, 4), input_shape=self.config.shape),
            Conv2D(16, kernel_size=(4, 4)),
            Flatten(),
            Dense(16),
            Dense(8),
            Activation('softmax'),
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        return model