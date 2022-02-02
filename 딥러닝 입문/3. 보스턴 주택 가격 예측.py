import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.datasets.boston_housing import load_data

(x_train, y_train), (x_test, y_test) = load_data(path='boston_housing.npz',
                                                 test_split=0.2,
                                                 seed=777)

x_train.shape
x_test.shape

# 데이터 전처리 및 검증 데이터 셋 만드릭
import numpy as np

mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train = (x_train - mean)/std
x_test = (x_test - mean)/std

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.33,
                                                  random_state = 777)

# 모델 구성하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])

# 학습하고 평가하기
history = model.fit(x_train, y_train, epochs = 300,
                    validation_data = (x_val, y_val))
model.evaluate(x_test, y_test)


# K-FOLD 사용하기
import numpy as np
from sklearn.model_selection import KFold

k = 3

kfold = KFold(n_splits=3, random_state=777, shuffle=True)

def get_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(13,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

mae_list = []

for train_index, val_index in kfold.split(x_train):
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = get_model()

    history = model.fit(x_train_fold, y_train_fold, epochs=300,
                        validation_data=(x_val_fold, y_val_fold))

    test_mae = model.evaluate(x_test, y_test)
    mae_list.append(test_mae[-1])


mae_list
np.mean(mae_list)



