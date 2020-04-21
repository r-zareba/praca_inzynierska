import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt


def load_csv(data_path: str, normalize=True) -> pd.DataFrame:

    column_names = ('time', 'u', 'i')
    df = pd.read_csv(data_path, delimiter=';', names=column_names, header=0)
    _normalize_df(df, normalize)
    return df


def _normalize_df(df: pd.DataFrame, normalize: bool) -> None:
    """
    Repalce ',' with '.', convert columns to float numbers
    and normalize to range (0, 1)
    """
    scaler = sklearn.preprocessing.MinMaxScaler() if normalize else None

    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].apply(lambda x: x.replace(',', '.'))
        df.loc[:, col] = df.loc[:, col].astype(np.float64)
        if normalize:
            df.loc[:, col] = scaler.fit_transform(
                df.loc[:, col].values.reshape(-1, 1))


def add_harmonic(arr: np.array, n: int, perc: float, phase: int) -> None:
    x = np.linspace(phase, (2 * n * np.pi) + phase, arr.shape[0])
    harmonic_signal = (np.sin(x) * perc).reshape(-1, 1)
    arr += harmonic_signal


def create_training_set(X: np.array, y: np.array, time_steps: int) -> tuple:
    x, y = [], []


path = '/Users/kq794tb/Desktop/Politechnika/praca_inzynierska/data/dane1.csv'
df = load_csv(path, normalize=True)

u1 = np.copy(df.loc[:, 'u'].values.reshape(-1, 1))
part = 5000

u2 = np.copy(u1[:part])
u3 = np.copy(u1[:part])
u4 = np.copy(u1[:part])
u5 = np.copy(u1[:part])
u6 = np.copy(u1[:part])
u7 = np.copy(u1[:part])
u8 = np.copy(u1[:part])
u9 = np.copy(u1[:part])

add_harmonic(u2, 3, 0.05, 0)
add_harmonic(u3, 3, 0.10, 200)
add_harmonic(u4, 3, 0.15, 0)
add_harmonic(u5, 3, 0.20, 500)
add_harmonic(u6, 11, 0.20, 0)
add_harmonic(u7, 7, 0.10, 200)
add_harmonic(u8, 5, 0.15, 0)
add_harmonic(u9, 11, 0.15, 0)
u = np.concatenate((u1, u2, u3, u4, u5, u6, u7, u8, u9))


i1 = np.copy(df.loc[:, 'i'].values.reshape(-1, 1))
i2 = np.copy(i1[:part])
i3 = np.copy(i1[:part])
i4 = np.copy(i1[:part])
i5 = np.copy(i1[:part])
i6 = np.copy(i1[:part])
i7 = np.copy(i1[:part])
i8 = np.copy(i1[:part])
i9 = np.copy(i1[:part])

add_harmonic(i2, 3, 0.05, 0)
add_harmonic(i3, 3, 0.10, 200)
add_harmonic(i4, 3, 0.15, 0)
add_harmonic(i5, 3, 0.20, 500)
add_harmonic(i6, 11, 0.20, 0)
add_harmonic(i7, 7, 0.10, 200)
add_harmonic(i8, 5, 0.15, 0)
add_harmonic(i9, 11, 0.15, 0)
i = np.concatenate((i1, i2, i3, i4, i5, i6, i7, i8, i9))


fig, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.plot(u)
ax1.set_title('u(t)')

ax2.plot(i)
ax2.set_title('i(t)')

plt.show()


