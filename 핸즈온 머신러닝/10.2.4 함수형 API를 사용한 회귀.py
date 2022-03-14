from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, random_state=271
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 와이드 딥 신경망
from tensorflow import keras
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model_1 = keras.Model(inputs=[input_], outputs=[output])

# 다중 입력
input_A = keras.layers.Input(shape=[4], name='wide_input')
input_B = keras.layers.Input(shape=[4], name='deep_input')
hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1, name='output')(concat)
model_2 = keras.Model(inputs=[input_A, input_B], outputs=[output])
model_2.summary()

model_2.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :4], X_train[:,4:]
X_test_A, X_test_B = X_test[:, :4], X_test[:,4:]

history = model_2.fit((X_train_A, X_train_B), y_train, epochs=30,
                    validation_data = ((X_test_A, X_test_B), y_test))

# 다중 입력 다중 출력
input_A = keras.layers.Input(shape=[4], name='wide_input')
input_B = keras.layers.Input(shape=[4], name='deep_input')
hidden1 = keras.layers.Dense(30, activation='relu')(input_B)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_A, hidden2])
output = keras.layers.Dense(1, name='output')(concat)
aux_model = keras.layers.Dense(1, name='aux_output')(hidden2)
model_3 = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_model])

# 보조 출력은 규제로만 사용되기 대문에 손실 가중치를 줄임
model_3.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer='sgd')

history = model_3.fit([X_train_A, X_train_B], [y_train, y_train], epochs=30,
                    validation_data = ([X_test_A, X_test_B], [y_test, y_test]))

total_loss, main_loss, aux_loss = model_3.evaluate([X_test_A, X_test_B], [y_test, y_test])

# 모델 저장과 복원
model_3.save('model_3.h5')
model_4 = keras.models.load_model('model_3.h5')
model_4.summary()


# 콜백 사용하기
import keras.callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint('model_3.h5', save_best_only=True)
history = model_3.fit([X_train_A, X_train_B], [y_train, y_train], epochs=30,
                    validation_data = ([X_test_A, X_test_B], [y_test, y_test]), callbacks=[checkpoint_cb])

model = keras.models.load_model('model_3')


# 조기 종료
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history = model_3.fit([X_train_A, X_train_B], [y_train, y_train], epochs=200,
                    validation_data = ([X_test_A, X_test_B], [y_test, y_test]),
                      callbacks=[checkpoint_cb, early_stopping_cb])

# 텐서 보드
import os
root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir,run_id)

run_logdir = get_run_logdir()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model_3.fit([X_train_A, X_train_B], [y_train, y_train], epochs=200,
                    validation_data = ([X_test_A, X_test_B], [y_test, y_test]),
                      callbacks=[checkpoint_cb, early_stopping_cb])
