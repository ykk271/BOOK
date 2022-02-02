from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train.shape
y_train.shape
x_test.shape

# 데이터 그려보기
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(777)

class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse',
               'sheep', 'truck']

sample_size = 9
random_idx = np.random.randint(60000, size=sample_size)

plt.figure(figsize=(5, 5))
for i, idx in enumerate(random_idx):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap='gray')
    plt.xlabel(class_names[int(y_train[i])])

plt.show()

# 전처리
x_mean = np.mean(x_train, axis=(0, 1, 2))
x_std = np.std(x_train, axis=(0, 1, 2))

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.3,
                                                  random_state=777)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding='same',
                 activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=32, kernel_size=3, padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                 activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.summary()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


# 신경망 시각화해보기
'''
import tensorflow as tf
get_layer_name = [layer.name for layer in model.layers]
get_output = [layer.output for layer in model.layers]

visual_model = tf.keras.models.Model(inputs = model.input, outputs= get_output)

test_img = np.expand_dims(x_test[i], axis=0)
feature_maps = visual_model.predict(test_img)

for layer_name, feature_map in zip()
'''

# 규제화 함수 사용하기, 드롭아웃, 배치 정규화
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

model2 = Sequential()
filterSize = 32
model2.add(Conv2D(filters=filterSize, kernel_size=3, padding='same', input_shape=(32, 32, 3)))
model2.add(BatchNormalization())
model2.add(Conv2D(filters=filterSize, kernel_size=3, padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model2.add(Dropout(0.2))

filterSize = 64
model2.add(Conv2D(filters=filterSize, kernel_size=3, padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(filters=filterSize, kernel_size=3, padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model2.add(Dropout(0.2))

filterSize = 128
model2.add(Conv2D(filters=filterSize, kernel_size=3, padding='same'))
model2.add(BatchNormalization())
model2.add(Conv2D(filters=filterSize, kernel_size=3, padding='same'))
model2.add(BatchNormalization())
model2.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model2.add(Dropout(0.2))


model2.add(Flatten())
model2.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model2.add(Dense(10, activation='softmax'))

model2.compile(optimizer=Adam(1e-4),
               loss='sparse_categorical_crossentropy',
               metrics=['acc'])

'''
history2 = model2.fit(x_train, y_train,
                      epochs=10,
                      batch_size=32,
                      validation_data=(x_val, y_val))
'''


# 데이터 증식 사용
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(horizontal_flip = True,
                                  zoom_range=0.2,
                                  width_shift_range= 0.1,
                                  height_shift_range=0.1,
                                  rotation_range=30,
                                  fill_mode='nearest')


val_datagen = ImageDataGenerator()
batch_size = 32

train_generator = train_datagen.flow(x_train, y_train,
                                      batch_size = batch_size)
val_generator = val_datagen.flow(x_val, y_val,
                                batch_size = batch_size)


def get_step(train_len, batch_size):
    if(train_len % batch_size >0):
        return train_len // batch_size + 1
    else:
        return train_len // batch_size

history2 = model2.fit(train_generator,
                    epochs=10,
                    steps_per_epoch=get_step(len(x_train), batch_size),
                    validation_data = val_generator,
                    validation_steps = get_step(len(x_val), batch_size))



# 전이 사용하기 - model3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

VGG16 = VGG16(weights = 'imagenet',
              input_shape = (32,32,3), include_top=False)
# 모델 동결 해제
for layer in VGG16.layers[:-4]:
    layer.trainable = False

model3 = Sequential()
model3.add(VGG16)
model3.add(Flatten())
model3.add(Dense(256))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Dense(10, activation = 'softmax'))

model3.summary()

model3.compile(optimizer=Adam(1e-4),
               loss='sparse_categorical_crossentropy',
               metrics=['acc'])

history3 = model3.fit(train_generator,
                      epochs=10,
                      steps_per_epoch=get_step(len(x_train), batch_size),
                      validation_data=val_generator,
                      validation_steps=get_step(len(x_val), batch_size))


# 학습과정 그려보기
import matplotlib.pyplot as plt

his_dict = history.history
acc = his_dict['acc']
val_acc = his_dict['val_acc']

his_dict2 = history2.history
acc2 = his_dict2['acc']
val_acc2 = his_dict2['val_acc']

his_dict3 = history3.history
acc3 = his_dict3['acc']
val_acc3 = his_dict3['val_acc']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 5))
ax1= fig.add_subplot(1,3,1)
ax1.plot(epochs, acc, color='blue', label='train_acc')
ax1.plot(epochs, val_acc, color='red', label='val_acc')
plt.legend()

ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(epochs, acc2, color='blue', label='train_acc')
ax2.plot(epochs, val_acc2, color='red', label='val_acc')
plt.legend()
plt.show()


ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(epochs, acc3, color='blue', label='train_acc')
ax3.plot(epochs, val_acc3, color='red', label='val_acc')
plt.legend()
plt.show()




