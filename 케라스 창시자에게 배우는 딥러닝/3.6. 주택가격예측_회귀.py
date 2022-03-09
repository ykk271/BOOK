from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

train_data.shape
test_data.shape

import numpy as np
# 정규화
train_data = (train_data - np.mean(train_data,axis=0)) / np.std(train_data,axis=0)
test_data = (test_data - np.mean(test_data,axis=0)) / np.std(test_data,axis=0)

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmspop', loss='mse', metrics=['mae'])
    return model

K = 4
num_val_samples = len(train_data)//K # // 나눗셈 후 소수점 버림
num_val_samples

num_epochs = 100

fir i in range(K):

