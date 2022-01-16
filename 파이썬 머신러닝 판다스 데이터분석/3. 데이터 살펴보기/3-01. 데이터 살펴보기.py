import pandas as pd

df = pd.read_csv('C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part3/auto-mpg.csv', header = None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

print(df.head())
print(df.tail())
print(df.shape)

print(df.info())

print(df.dtypes)
print(df.mpg.dtypes)

print(df.describe())

