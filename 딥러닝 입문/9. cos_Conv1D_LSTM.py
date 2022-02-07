# 데이터 생성
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2022)

time = np.arange(30*12 + 1)
time
month_time = (time%30)/30
month_time
time_series = 20 * np.where(month_time < 0.5,
                            np.cos(2 * np.pi * month_time),
                            np.cos(2 * np.pi * month_time) + np.random.random(361))
time_series

def make_sequence(time_series, n):
    x_train, y_train = list(), list()

    for i in range(len(time_series)):
        x = time_series[i:(i+n)]
        if (i + n) < len(time_series):
            x_train.append(x)
            y_train.append(time_series[i+n])
        else:
            break
    return np.array(x_train), np.array(y_train)

n = 10
x_train, y_train = make_sequence(time_series, n)
x_train = x_train.reshape(-1,n,1)
y_train = y_train.reshape(-1, 1)

from sklearn.model_selection import train_test_split

patial_x_train = x_train[:30*11]
patial_y_train = y_train[:30*11]

x_test = x_train[30*11:]
y_test = y_train[30*11:]

patial_x_train.shape
patial_y_train.shape
x_test.shape

# 모델 구성 및 확인
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential()
model.add(Conv1D(32, 3, activation = 'relu', input_shape = (10,1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(32, 3, activation='relu'))
model.add(LSTM(32, dropout=0.2, recurrent_dropout = 0.2))
model.add(Dense(1))

model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics = ['mse'])

model.fit(patial_x_train, patial_y_train, epochs = 100, batch_size = 32)
pred = model.predict(x_test)
pred


train_range = np.arange(len(x_train) + 1)
pred_range = np.arange(len(y_train), len(y_train)+len(pred))


plt.plot(train_range, np.append(y_train, y_test[0]), color = 'black')
plt.plot(pred_range, y_test, color = 'orange')
plt.plot(pred_range, pred, color = 'blue')