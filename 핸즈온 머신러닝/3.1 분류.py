from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist['data'], mnist['target']
X.shape

import matplotlib.pyplot as plt

import random
random_num = random.randint(0, 69999)

some_digit = X.iloc[random_num]
some_digit.shape
import numpy as np
some_digit = np.array(some_digit)
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.title(y[random_num])

y[0]
type(y[0])

y = y.astype(np.uint8)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, shuffle = True,
                                                stratify=y, random_state=271)


X_train.shape

import pandas as pd
train_count = pd.value_counts(y_train)
train_count.plot.bar()

# 이진 분류기 훈련
y_train_5 = (y_train == 5)
y_val_5 = (y_val == 5)

# 확률적 경사 하강법 Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=271)
sgd_clf.fit(X_train, y_train_5)

# 성능 측정
