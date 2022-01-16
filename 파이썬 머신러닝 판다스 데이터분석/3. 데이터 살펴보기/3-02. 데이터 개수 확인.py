import pandas as pd

df = pd.read_csv('C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part3/auto-mpg.csv', header = None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

print(df.count())
print(type(df.count()))

unique_value = df['origin'].value_counts()
print(unique_value)
print(type(unique_value))