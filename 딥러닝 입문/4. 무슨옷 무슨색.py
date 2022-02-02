import pandas as pd

DATA_PATH = 'D:/KAGGLE DATA/apparel image dataset'

train_df = pd.read_csv(DATA_PATH+'/train.csv')
val_df = pd.read_csv(DATA_PATH+'/val.csv')
test_df = pd.read_csv(DATA_PATH+'/test.csv')


pd.set_option('display.max_columns', 10)
train_df.head()
pd.set_option('display.width', 10)
train_df.columns

# 이미지 제너레이터 정의 및 모델 구성하기
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0:
        return (num_samples // batch_size) + 1
    else:
        return num_samples // batch_size


# 모델 구성하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(Flatten(input_shape = (112, 112, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation='sigmoid'))

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['acc'])

# 데이터 제네레이터 정의
batch_size = 32
class_col = ['black', 'blue', 'brown', 'green', 'red', 'white',
             'dress', 'shirt', 'pants', 'shorts', 'shoes']

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_PATH,
    x_col='image',
    y_col=class_col,
    target_size=(112,112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    seed=42)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=DATA_PATH,
    x_col='image',
    y_col=class_col,
    target_size=(112,112),
    color_mode='rgb',
    class_mode='raw',
    batch_size=batch_size,
    shuffle=True
)

# 모델 학습
model.fit(train_generator,
          steps_per_epoch=get_steps(len(train_df), batch_size),
          validation_data = val_generator,
          validation_steps=get_steps(len(val_df), batch_size),
          epochs=10
          )

