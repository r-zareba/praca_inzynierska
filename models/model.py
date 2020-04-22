import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Encoder(layers.Layer):
    def __init__(self, n_neurons: int, n_timestamps: int, n_features: int):
        super().__init__()
        self._lstm1 = layers.Bidirectional(
            layers.LSTM(units=n_neurons, activation='relu',
                        input_shape=(n_timestamps, n_features),
                        return_sequences=True))
        self._lstm2 = layers.Bidirectional(
            layers.LSTM(units=int(n_neurons / 2), activation='relu',
                        return_sequences=False))

    def call(self, x):
        x = self._lstm1(x)
        return self._lstm2(x)


class Decoder(layers.Layer):
    def __init__(self, n_neurons: int, n_timestamps: int, n_features: int):
        super().__init__()
        self._lstm1 = layers.Bidirectional(
            layers.LSTM(units=int(n_neurons / 2), activation='relu',
                        input_shape=(n_timestamps, n_features),
                        return_sequences=True))
        self._lstm2 = layers.Bidirectional(
            layers.LSTM(units=n_neurons, activation='relu',
                        return_sequences=True))

    def call(self, x):
        x = self._lstm1(x)
        return self._lstm2(x)


class Model(tf.keras.Model):
    def __init__(self, n_neurons: int, n_timestamps: int, n_features: int):
        super().__init__()
        self._encoder = Encoder(n_neurons, n_timestamps, n_features)
        self._decoder = Decoder(n_neurons, n_timestamps, n_features)

        self._repeat_vector = layers.RepeatVector(n_timestamps)
        self._time_distributed = layers.TimeDistributed(
            layers.Dense(n_features))

    def call(self, x):
        x = self._encoder(x)
        x = self._repeat_vector(x)
        x = self._decoder(x)
        return self._time_distributed(x)

    def calculate_reconstruction_loss(self, x: np.array) -> np.array:
        y_pred = self.predict(x)
        loss = np.mean((x - y_pred) ** 2, axis=1)
        return np.mean(loss, axis=1)

    def calculate_anomaly_threshold(self, loss: np.array) -> float:
        return loss.max() + (0.25 * loss.std())
