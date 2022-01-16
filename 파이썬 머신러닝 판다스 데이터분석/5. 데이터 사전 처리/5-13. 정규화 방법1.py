import pandas as pd
import numpy as np

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df['horsepower'].replace('?', np.nan, inplace=True)  
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

# max 확인
print(df.horsepower.describe())

df.horsepower = df.horsepower/abs(df.horsepower.max())

print(df.horsepower.head())
print(df.horsepower.describe())