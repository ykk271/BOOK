from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data.shape

a = range(1, train_data.shape[0])

for i in a:
    print('{0} {1}'.format(i, len(train_data[i])))

# 데이터 준비
import numpy as np

train_data[0]
max([max(sequence) for sequence in train_data])
[max(sequence) for sequence in train_data]

word_index = imdb.get_word_index()
reverse_word_index = dict([value, key] for (key, value) in word_index.items())
reverse_word_index

decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[1]])
decoded_review


def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 모델 만들기
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 훈련 검증

from sklearn.model_selection import train_test_split
x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size = 0.3)


# 모델 훈련하기
history = model.fit(x_train_new,
                    y_train_new,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))



# 훈련과 검증 손실 그리기
import pandas as pd
import matplotlib.pyplot as plt
a = pd.DataFrame(history.history)
plt.plot(a)
plt.legend(a.columns)
plt.show()

results = model.evaluate(x_test, y_test)
print(results)

