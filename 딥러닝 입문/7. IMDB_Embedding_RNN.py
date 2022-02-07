from tensorflow.keras.datasets import imdb

num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train.shape
x_test.shape
y_test.shape

x_train[0]
y_train[0] # 긍정 1, 부정 0

# 가장 빈번하게 쓰는 단어
imdb_get_word_index = {}

for key, value in imdb.get_word_index().items():
    imdb_get_word_index[value]  = key

imdb_get_word_index[1]
imdb_get_word_index[2]

# 데이터를 동일한 길이로 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 500
pad_x_train = pad_sequences(x_train, maxlen = max_len, padding='pre')
pad_x_test = pad_sequences(x_test, maxlen = max_len, padding='pre')

len(x_train[0])
len(pad_x_train[0])
x_train[0]

imdb.get_word_index('a')

# 모델 구성하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

model = Sequential()
model.add(Embedding(input_dim = num_words, output_dim=32, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history = model.fit(pad_x_train, y_train, batch_size=32, epochs=30,
                    validation_split=0.2)

model.evaluate(pad_x_test, y_test)

# Simple RNN
from tensorflow.keras.layers import SimpleRNN, Flatten, Dense
from tensorflow.keras.models import Sequential

model2 = Sequential()
model2.add(Embedding(input_dim = num_words, output_dim=32, input_length=max_len))
model2.add(SimpleRNN(32, return_sequences = True, dropout = 0.15,
                     recurrent_dropout=0.15))
model2.add(SimpleRNN(32))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='adam',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

model2.summary()

history2 = model2.fit(pad_x_train, y_train, batch_size=32, epochs=30,
                    validation_split=0.2)

import matplotlib.pyplot as plt

his_dict = history.history
acc = his_dict['acc']
val_acc = his_dict['val_acc']

his_dict2 = history2.history
acc2 = his_dict2['acc']
val_acc2 = his_dict2['val_acc']

epochs = range(1, len(acc) + 1)

fig = plt.figure(figsize=(10, 5))
ax1= fig.add_subplot(1,2,1)
ax1.plot(epochs, acc, color='blue', label='train_acc')
ax1.plot(epochs, val_acc, color='red', label='val_acc')
plt.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, acc2, color='blue', label='train_acc')
ax2.plot(epochs, val_acc2, color='red', label='val_acc')
plt.legend()
plt.show()

