import os, shutil

train_data_dir = 'D:/KAGGLE DATA/dogs-vs-cats/train/train'
len(os.listdir(train_data_dir))

'''
##### 데이터 준비 ##
train_data_cats_dir = os.path.join(train_data_dir, 'cats')
os.mkdir(train_data_cats_dir)

train_data_dogs_dir = os.path.join(train_data_dir, 'dogs')
os.mkdir(train_data_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(12500)]
for fname in fnames:
    C = os.path.join(train_data_dir, fname)
    V = os.path.join(train_data_cats_dir, fname)
    print(fname)
    shutil.copyfile(C, V)

fnames = ['dog.{}.jpg'.format(i) for i in range(12500)]
for fname in fnames:
    C = os.path.join(train_data_dir, fname)
    V = os.path.join(train_data_dogs_dir, fname)
    shutil.copyfile(C, V)
###############################
'''


# 모델
import tensorflow.keras
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])


# imgDataGen
from keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_data_gen = ImageDataGenerator(
    rescale=1./255,
)


# train_data_dir = 'D:/KAGGLE DATA/dogs-vs-cats/cats_and_dog_small/train'
val_data_dir = 'D:/KAGGLE DATA/dogs-vs-cats/cats_and_dog_small/validation'

train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


val_generator = val_data_gen.flow_from_directory(
    val_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=64,
    epochs=10,
    validation_data=val_generator,
    validation_steps=10
)

model.save('cats_and_dogs.h5')

