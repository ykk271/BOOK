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


