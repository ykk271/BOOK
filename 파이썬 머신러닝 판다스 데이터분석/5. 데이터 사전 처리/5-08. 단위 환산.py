import pandas as pd

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

print(df.head(3))

mpg_to_kpl = 1.609/3.785

df['kpl'] = df['mpg'] * mpg_to_kpl

print(df.head(3))

df['kpl'] = df['kpl'].round(2)
print(df.head(3))

