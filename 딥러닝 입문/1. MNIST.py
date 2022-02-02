# 데이터
import matplotlib
from tensorflow.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')
x_train.shape
x_test.shape


# 데이터 그려보기
import matplotlib.pyplot as plt
import numpy as np

sample_size = 3
random_idx = np.random.randint(60000, size=sample_size)
random_idx

for idx in random_idx:
    img = x_train[idx, :]
    label = y_train[idx]
    plt.figure()
    plt.imshow(img)
    plt.title('%d-th data, label is %d' % (idx, label))


# 검증 데이터 만들기
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size = 0.3,
                                                  random_state = 777)
x_val.shape
y_val.shape


# 모델 입력을 위한 데이터 전처리
num_x_train = x_train.shape[0]
num_x_val = x_val.shape[0]
num_x_test = x_test.shape[0]
num_x_train

x_train = (x_train.reshape((num_x_train, 28*28))) / 255
x_val = (x_val.reshape((num_x_val, 28*28))) / 255
x_test = (x_test.reshape((num_x_test, 28*28))) / 255
x_train.shape

# 모델 입력을 위한 레이블 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
y_train[0]

# 모델 구성하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape=(784, )))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()

# 학습과정 설정하기
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(x_train, y_train,
                    epochs = 30,
                    batch_size=128,
                    validation_data=(x_val, y_val))

# history를 통해 확인해볼 수 있는 값 출력
history.history.keys()

# 학습결과 그려보기
import matplotlib.pyplot as plt

his_dict = history.history
acc = his_dict['acc']
val_acc = his_dict['val_acc']

epochs = range(1, len(acc)+1)
fig = plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, color='blue', label='train_acc')
plt.plot(epochs, val_acc, color='red', label='val_acc')
plt.legend()
plt.show()

# 모델 평가하기
model.evaluate(x_test, y_test)

# 학습된 모델을 통해 값 예측하기
import numpy as np

results = model.predict(x_test)
results.shape
np.set_printoptions(precision=1) # numpy 소수점 제한
print(f'각 클래스에 속학 확률: \n{results[0]}')

# 예측값 그려서 확인해보기
import matplotlib.pyplot as plt
arg_results = np.argmax(results, axis=-1)
plt.imshow(x_test[0].reshape(28, 28))
plt.title('Precicted value of the first image: ' + str(arg_results[0]))
plt.show()

# 모델 평가 방법-1 혼동행렬
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

# 모델 평가 방법-2 분류 보고서
print('\n', classification_report(np.argmax(y_test, axis=-1),
                                  np.argmax(results, axis=-1)))


