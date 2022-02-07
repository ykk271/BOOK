# Sequential() 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.utils import plot_model

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(strides = 2))
model.add(GlobalMaxPooling2D())
model.add(Dense(1, activation = 'sigmoid'))

import graphviz
import pydot
import pydotplus

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plot_model(model)

# 함수형 API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.layers import Input

inputs = Input(shape = (224, 224, 3))
x = Conv2D(32, (3,3), activation = 'relu')(inputs)
x = Conv2D(32, (3,3), activation = 'relu')(x)
x = MaxPooling2D(strides = 2)(x)
x = GlobalMaxPooling2D()(x)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = inputs, outputs = x)

