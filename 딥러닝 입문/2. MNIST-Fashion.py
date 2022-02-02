# 데이터셋 다운받기
from tensorflow.keras.datasets.fashion_mnist import load_data
(x_train, y_train), (x_test, y_test) = load_data()
x_train.shape

# 데이터 그려보기
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(777)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

sample_size = 9
random_idx = np.random.randint(60000, size=sample_size)

plt.figure(figsize=(5, 5))
for i, idx in enumerate(random_idx):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap='gray')
    plt.xlabel(class_names[y_train[i]])
plt.show()

# 전처리 및 검증 데이터 셋 만들기
x_train = x_train/255
x_test = x_test/255

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.3,
                                                  random_state=777)

x_train.shape
x_val.shape

# 첫 번째 모델 구성하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

first_model = Sequential()
first_model.add(Flatten(input_shape = (28,28)))
first_model.add(Dense(64, activation = 'relu'))
first_model.add(Dense(32, activation = 'relu'))
first_model.add(Dense(10, activation = 'softmax'))

# 학습 과정 설정 및 학습하기
first_model.compile(optimizer='adam',
                    loss = 'categorical_crossentropy',
                    metrics=['acc'])
first_history = first_model.fit(x_train, y_train,
                                epochs = 30,
                                batch_size=128,
                                validation_data = (x_val, y_val))


# 두 번째 모델 구성 - 더 깊은 Dense 층
Second_model = Sequential()
Second_model.add(Flatten(input_shape = (28,28)))
Second_model.add(Dense(128, activation = 'relu'))
Second_model.add(Dense(128, activation = 'relu'))
Second_model.add(Dense(64, activation = 'relu'))
Second_model.add(Dense(32, activation = 'relu'))
Second_model.add(Dense(10, activation = 'softmax'))

Second_model.compile(optimizer='adam',
                    loss = 'categorical_crossentropy',
                    metrics=['acc'])
Second_history = Second_model.fit(x_train, y_train,
                                epochs = 40,
                                batch_size=128,
                                validation_data = (x_val, y_val))

# 두 모델의 학습과정 그려보기
import numpy as np
import matplotlib.pyplot as plt

def draw_acc(history_1, history_2):
    his_dict_1 = history_1.history
    his_dict_2 = history_2.history
    acc_1 = his_dict_1['acc']
    val_acc_1 = his_dict_1['val_acc']
    acc_2 = his_dict_2['acc']
    val_acc_2 = his_dict_2['val_acc']

    epochs_1 = range(1, len(acc_1) + 1)
    epochs_2 = range(1, len(acc_2) + 1)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(epochs_1, val_acc_1, color='blue', label='train_acc')
    plt.plot(epochs_2, val_acc_2, color='red', label='val_acc')
    plt.legend()
    plt.show()

draw_acc(first_history, Second_history)
# 더 깊다고 성능이 좋아지는 건 아니고 과대적합 문제에 빠질 수 있음

# 모델 평가하기
first_model.evaluate(x_test, y_test)
Second_model.evaluate(x_test, y_test)

results = first_model.predict(x_test)

import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(7,7))
cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(results, axis=-1))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

