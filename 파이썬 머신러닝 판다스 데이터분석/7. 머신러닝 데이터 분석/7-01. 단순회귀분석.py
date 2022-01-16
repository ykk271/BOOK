import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part7/'

df = pd.read_csv(data_path + 'auto-mpg.csv')

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model year', 'origin', 'mean']

print(df.head())

pd.set_option('display.max_columns', 50)
print(df.head())

print(df.info())

print(df.describe())

print(df['horsepower'].unique())

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]
print(ndf.head())

ndf.plot(kind='scatter', x='weight', y='mpg', c = 'coral', s=10, figsize=(10,3))
plt.show()

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax1) # 회귀선 표시
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax2, fit_reg=False) # 회귀선 미표시
plt.show()

grid_ndf = sns.pairplot(ndf)
plt.show()

x = ndf[['weight']]
y = ndf[['mpg']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=10)

print('train data 개수: ', len(x_train))
print('test data 개수: ', len(x_test))

from sklearn.linear_model import LinearRegression

# 단순회귀분석 모형 객체 생성
lr = LinearRegression()

lr.fit(x_train, y_train)

r_square = lr.score(x_test, y_test)
print(r_square)

print('기울기 a:', lr.coef_)
print('y 절편 b', lr.intercept_)

y_hat = lr.predict(x)

plt.figure(figsize=(10, 5))
ax1 = sns.distplot(y, hist=False, label="Y")
ax2 = sns.distplot(y_hat, hist=False, label="Y_hat", ax=ax1)
plt.show()
