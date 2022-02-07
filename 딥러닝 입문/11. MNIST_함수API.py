from tensorflow.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.3,
                                                  random_state = 777)

x_train = (x_train.reshape(-1, 28, 28, 1))/255
x_val = (x_val.reshape(-1, 28, 28, 1))/255
x_test = (x_test.reshape(-1, 28, 28, 1))/255

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 구성 및 학습
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.layers import Input

inputs = Input(shape = (28, 28, 1))
x = Conv2D(32, (3,3), activation = 'relu')(inputs)
x = Conv2D(32, (3,3), activation = 'relu')(x)
x = MaxPooling2D(strides = 2)(x)
x = GlobalMaxPooling2D()(x)
x = Dense(10, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = x)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size = 32,
          validation_data=(x_val, y_val),
          epochs = 10)