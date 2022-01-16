import pandas as pd
import numpy as np

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df['horsepower'].replace('?', np.nan, inplace=True)  
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

print(df['horsepower'].sample(10))

# np.historam 함수로 3개의 bin으로 구분할 경계값 리스트 찾기
count, bin_dividers = np.histogram(df['horsepower'], bins=3)
print(bin_dividers)

bin_names = ['저출력', '보통출력', '고출력']

df['hp_bin'] = pd.cut(x=df['horsepower'],
                      bins=bin_dividers,
                      labels=bin_names,
                      include_lowest=True)  # 첫 경계값 포함

print(df[['horsepower', 'hp_bin']].head(15))

#horsepower_dummies = pd.get_dummies(df['hp_bin'])