import pandas as pd

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

print(df.dtypes)

print(df['horsepower'].unique())

import numpy as np

df['horsepower'].replace('?', np.nan, inplace = True) # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True) # 누락 데이터 행 삭제
df['horsepower'] = df['horsepower'].astype('float') # 문자열을 실수형으로 변환

print(df['horsepower'].dtypes)

print(df['origin'].unique())

df['origin'].replace({1:'USA', 2:'EU', 3:'JPN'}, inplace=True)

print(df['origin'].unique())
print(df['origin'].dtypes)

df['origin'] = df['origin'].astype('category')
print(df['origin'].dtypes)

df['origin'] = df['origin'].astype('str')
print(df['origin'].dtypes)

print(df['model year'].sample(3))
df['model year'] = df['model year'].astype('category')
print(df['model year'].sample(3))

