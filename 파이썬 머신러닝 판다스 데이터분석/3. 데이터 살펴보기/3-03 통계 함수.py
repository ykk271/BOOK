import pandas as pd

df = pd.read_csv('C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part3/auto-mpg.csv', header = None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

print(df.mean())
print(df['mpg'].mean())
print(df.mpg.mean())
print(df[['mpg','weight']].mean())

print(df.median())
print(df['mpg'].median())

print(df.max())
print(df['mpg'].max())

print(df.min())
print(df['mpg'].min())

print(df.std())
print(df['mpg'].std())

# 상관계수
print(df.corr())
print(df[['mpg','weight']].corr())

