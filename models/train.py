import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
import time

from model import Model

CSV_COLUMN_NAMES = ('time', 'u', 'i')


def load_csv(data_path: str, normalize=True) -> pd.DataFrame:
    df = pd.read_csv(data_path, delimiter=';', names=CSV_COLUMN_NAMES, header=0)
    _normalize_df(df, normalize)
    return df


def _normalize_df(df: pd.DataFrame, normalize: bool) -> None:
    """
    Repalce ',' with '.', convert columns to float type
    and normalize to range (0, 1)
    """
    scaler = sklearn.preprocessing.MinMaxScaler((0, 1)) if normalize else None

    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].apply(lambda x: x.replace(',', '.'))
        df.loc[:, col] = df.loc[:, col].astype(np.float64)
        if normalize:
            df.loc[:, col] = scaler.fit_transform(
                df.loc[:, col].values.reshape(-1, 1))


def add_harmonic(arr: np.array, n: int, perc: float, phase: int) -> None:
    x = np.linspace(phase, (2 * n * np.pi) + phase, arr.shape[0])
    harmonic_signal = (np.sin(x) * (perc / 100)).reshape(-1, 1)
    arr += harmonic_signal


path = '/Users/kq794tb/Desktop/Politechnika/praca_inzynierska/data/dane3.csv'
df = load_csv(path, normalize=True)

n_timestamps = 16
batch_size = 64
shuffle_batch = 1000

x = df.loc[:, ['i', 'u']].values

training_array = []
for i in range(len(x) - n_timestamps):
    training_array.append(x[i:i + n_timestamps])

x_train = np.array(training_array)
y_train = x_train

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.cache().batch(
    batch_size).shuffle(shuffle_batch).repeat()

model = Model(n_neurons=64,
              n_timestamps=n_timestamps,
              n_features=x_train.shape[-1])

model.compile(loss='mse', optimizer='adam')

start_time = time.time()

steps_per_epoch = x_train.shape[0] // batch_size
history = model.fit(train_dataset,
                    epochs=5,
                    steps_per_epoch=steps_per_epoch)


print(f'Training took {time.time() - start_time} seconds')


y_pred = model.predict(x_train)

loss = model.calculate_reconstruction_loss(x_train)
anomaly_threshold = model.calculate_anomaly_threshold(loss)

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(12, 12))

ax1.plot(x_train[:, 0, 0], label='True i(t)', alpha=0.7)
ax1.plot(y_pred[:, 0, 0], label='Predicted i(t)')
ax1.legend(loc='upper right', prop={'size': 12})

ax2.plot(x_train[:, 0, 1], label='True u(t)', alpha=0.7)
ax2.plot(y_pred[:, 0, 1], label='Predicted u(t)')
ax2.legend(loc='upper right', prop={'size': 12})

ax3.plot(loss, label='Reconstruction loss')
ax3.axhline(y=anomaly_threshold, label='Anomaly threshold', c='r', linestyle='dashed')
ax3.set_ylim([0, anomaly_threshold + (0.3 * anomaly_threshold)])
ax3.legend(loc='upper right', prop={'size': 12})

plt.show()


def add_harmonic(arr: np.array, n: int, perc: float, phase: int) -> None:
    x = np.linspace(phase, (2 * n * np.pi) + phase, arr.shape[0])
    harmonic_signal = (np.sin(x) * (perc / 100)).reshape(-1, 1)
    arr += harmonic_signal


""" Testing on corrupted data """
u1 = np.copy(df.loc[:, 'u'].values.reshape(-1, 1))
part = int(u1.shape[0] / 8)

u2 = np.copy(u1[:part])
u3 = np.copy(u1[:part])
u4 = np.copy(u1[:part])
u5 = np.copy(u1[:part])
u6 = np.copy(u1[:part])
u7 = np.copy(u1[:part])
u8 = np.copy(u1[:part])
u9 = np.copy(u1[:part])

add_harmonic(u2, 3, 2, 0)
add_harmonic(u3, 5, 2, 0)
add_harmonic(u4, 3, 4, 0)
add_harmonic(u5, 3, 4, 0)
add_harmonic(u6, 11, 5, 0)
add_harmonic(u7, 7, 6, 0)
add_harmonic(u8, 5, 4, 0)
add_harmonic(u9, 11, 4, 0)
voltage = np.concatenate((u1, u2, u3, u4, u5, u6, u7, u8, u9))
# voltage = u1


i1 = np.copy(df.loc[:, 'i'].values.reshape(-1, 1))
i2 = np.copy(i1[:part])
i3 = np.copy(i1[:part])
i4 = np.copy(i1[:part])
i5 = np.copy(i1[:part])
i6 = np.copy(i1[:part])
i7 = np.copy(i1[:part])
i8 = np.copy(i1[:part])
i9 = np.copy(i1[:part])

add_harmonic(i2, 3, 5, 0)
add_harmonic(i3, 5, 6, 0)
add_harmonic(i4, 3, 6, 0)
add_harmonic(i5, 3, 11, 0)
add_harmonic(i6, 11, 11, 0)
add_harmonic(i7, 5, 11, 0)
add_harmonic(i8, 7, 5, 0)
add_harmonic(i9, 11, 6, 0)
current = np.concatenate((i1, i2, i3, i4, i5, i6, i7, i8, i9))
# current = i1

x_real = np.concatenate((current, voltage), axis=1)

# x_real = X
Xs = []
ys = []
for i in range(len(x_real) - n_timestamps):
    Xs.append(x_real[i:i+n_timestamps])

x_real = np.array(Xs)
real_loss = model.calculate_reconstruction_loss(x_real)
#
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(voltage)
ax1.set_title('u(t)')

ax2.plot(current)
ax2.set_title('i(t)')

ax3.plot(real_loss)
ax3.set_title('Reconstruction loss')
ax3.axhline(y=anomaly_threshold, label='Anomaly threshold', c='r', linestyle='--')
ax3.legend(loc='upper right')


plt.show()
