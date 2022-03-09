from tensorflow import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape
x_test.shape

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_img = x_train.reshape(60000,28*28)
train_img.shape
train_img/255

test_img = x_test.reshape(x_test.shape[0],28*28)
test_img.shape
test_img/255

from tensorflow.keras.utils import to_categorical
train_y = to_categorical(y_train)
test_y  = to_categorical(y_test)

model.fit(train_img, train_y, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_img, test_y)
test_acc

