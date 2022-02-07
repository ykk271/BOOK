import numpy as np

# 햄버거 사진
ham_img = np.random.random((1000, 28, 28, 1))
# 햄버거에 대한 평가
customer_form = np.random.randint(10000, size = (1000, 100))
customer_form[0]

# 햄버거에 대한 평점
ham_rate = np.round(np.random.random((1000,))*5, 1)
# 햄버거에 추가되어질 50가지의 재료
update_for_ham = np.random.randint(50, size = (1000,))

ham_img.shape
customer_form.shape
ham_rate.shape
update_for_ham.shape
update_for_ham[0]


# 모델 구성하기
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.layers import Dense, Concatenate

img_input = Input(shape = (28,28,1), name = 'ham_img')
form_input = Input(shape = (None, ), name = 'customer_form')

# 햄버거 이미지 입력
x_1 = Conv2D(32, (3, 3), activation = 'relu')(img_input)
x_1 = Conv2D(32, (3, 3), activation = 'relu')(x_1)
x_1 = MaxPooling2D(strides = 2)(x_1)
x_1 = GlobalMaxPooling2D()(x_1)

# 햄버거에 대한 평가 입력
x_2 = Embedding(10000, 64)(form_input)
x_2 = LSTM(128)(x_2)

x = Concatenate()([x_1, x_2])

# 햄버거 평점에 대한 출력값
rate_pred = Dense(1, name = 'ham_rate')(x)
# 50가지 재료에 대한 출력값
update_pred = Dense(50, activation='softmax',
                    name='update_for_ham')(x)

model = Model(inputs = [img_input, form_input],
              outputs = [rate_pred, update_pred])

from tensorflow.keras.utils import plot_model
plot_model(model)

model.compile(optimizer = 'adam',
              loss = ['mse', 'sparse_categorical_crossentropy'],
              metrics = {'ham_rate':'mse', 'update_for_ham':'acc'})

# 또는
model.compile(optimizer = 'adam',
              loss = {'ham_rate':'mse',
                      'update_for_ham':'sparse_categorical_crossentropy'},
              metrics = {'ham_rate':'mse', 'update_for_ham':'acc'})

# 학습
model.fit([ham_img, customer_form],
          [ham_rate, update_for_ham],
          epochs=2, batch_size=32)
