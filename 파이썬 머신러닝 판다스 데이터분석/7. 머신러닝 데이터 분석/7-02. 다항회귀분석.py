import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part7/'

df = pd.read_csv(data_path + 'auto-mpg.csv')

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model year', 'origin', 'mean']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]

x = ndf[['weight']]
y = ndf['mpg']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

print('훈련 데이터: ', x_train.shape)
print('검증 데이터: ', x_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # 다항식 변환

poly = PolynomialFeatures(degree = 2)
x_trian_ploy = poly.fit_transform(x_train) # x_train 데이터를 2차항으로 변형

print('원 데이터: ', x_train.shape)
print('2차항 변환 데이터: ', x_trian_ploy.shape)

pr = LinearRegression()
pr.fit(x_trian_ploy, y_train)

# 결정계수 계산
x_test_poly = poly.fit_transform(x_test)
r_square = pr.score(x_test_poly, y_test)
print(r_square)

y_hat_test = pr.predict(x_test_poly)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)
ax.plot(x_train, y_train, 'o', label='Train Data')
ax.plot(x_test, y_hat_test, 'r+', label='Predicted Value')
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
plt.close()

x_poly = poly.fit_transform(x)
y_hat = pr.predict(x_poly)

plt.figure(figsize=(10,5))
ax1 = sns.distplot(y, hist=False, label="Y")
ax2 = sns.distplot(y_hat, hist=False, label="Y hat", ax=ax1)
plt.show()
plt.close()