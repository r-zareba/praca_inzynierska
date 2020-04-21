import tensorflow as tf
from tensorflow.keras.layers import Dense


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._dense_1 = Dense(8)
        self._dense_2 = Dense(2)

    def call(self, x, y):

        x = self._dense_1(x)
        x = self._dense_2(y)

        return x