# 데이터셋 다운받기
from tensorflow.keras.datasets.fashion_mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data()
x_train.shape

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

import numpy as np

x_train = np.reshape(x_train/255, (-1, 28, 28, 1))
x_test = np.reshape(x_test/255, (-1, 28, 28, 1))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.3,
                                                  random_state=777)


# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

model = Sequential([
    Conv2D(filters = 16, kernel_size = 3, padding = 'same',
           activation='relu', input_shape=(28,28,1)),
    MaxPool2D(pool_size = (2,2), strides = 2, padding = 'same'),
    Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'),
    MaxPool2D(pool_size = (2,2), strides = 2, padding = 'same'),
    Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'),
    MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics=['acc'])

model.fit(x_train, y_train,
          epochs=30,
          batch_size = 128,
          validation_data = (x_val, y_val))

'''
from tensorflow.keras.utils import plot_model
import pydot
import graphviz
plot_model(model, './model.phg', show_shapes=True)
'''
