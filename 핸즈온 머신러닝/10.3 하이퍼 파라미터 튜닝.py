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

from tensorflow import keras
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(1))

def