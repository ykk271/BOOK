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
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

K = 4
num_val_samples = len(train_data)//K # // 나눗셈 후 소수점 버림
num_val_samples

num_epochs = 100
all_scores=[]

for i in range(K):
    print("처리중인 폴드 #", i)
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]

    partial_tarin_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0
    )
    partial_tarin_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0
    )

    model = build_model()
    model.fit(partial_tarin_data, partial_tarin_targets, epochs=num_epochs,
              batch_size=128)
    val_mse, val_mae = model.evaluate(val_data, val_targets,verbose=0)
    all_scores.append(val_mae)



all_scores