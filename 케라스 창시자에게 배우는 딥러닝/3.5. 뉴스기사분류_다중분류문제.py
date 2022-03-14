import pandas as pd
import tensorflow.keras
from keras.datasets import reuters

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=10000)
a = len(X_train)
b = len(X_test)
b/(a+b)

X_train.shape

X_train[1]

y_train

import numpy as np
np.unique(y_train)

def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

X_train_new = vectorize_sequences(X_train)
X_test_new = vectorize_sequences(X_test)

import pandas as pd
y_train_new = pd.get_dummies(y_train)
y_test_new = pd.get_dummies(y_test)

# y_train_new.to_numpy()
y_train.shape
y_train_new.shape

'''
from keras.utils.np_utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

y_train_one_hot.shape
'''

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 훈련과 그림 1
history = model.fit(X_train_new,
                    y_train_new,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (X_test_new, y_test_new))
import pandas as pd
import matplotlib.pyplot as plt
a = pd.DataFrame(history.history)
plt.plot(a)
plt.legend(a.columns)
plt.show()


