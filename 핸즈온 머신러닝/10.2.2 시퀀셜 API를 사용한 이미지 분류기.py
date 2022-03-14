import tensorflow as tf
from tensorflow import keras
tf.__version__
keras.__version__

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_train_full.shape
X_train_full.dtype

X_train_full[1]

X_train_full = X_train_full / 255.0
X_test =  X_test / 255.0

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, shuffle=True, stratify=y_train_full, random_state=271)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names[y_train[1]]

# 모델 만들기
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

## 참고
model.layers
model.layers[1].name
weights, biases = model.layers[1].get_weights()
weights
biases

model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)

# 예측하기
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)

# tf 2.6이후로 predict_classes가 아니라 predict
y_pred = model.predict(X_new)
predicted = y_pred.argmax(axis=-1)

import numpy as np
np.array(class_names)[predicted]

