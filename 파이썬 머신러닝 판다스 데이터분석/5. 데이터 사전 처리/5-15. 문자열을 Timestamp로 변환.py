import pandas as pd

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'stock-data.csv')

print(df.head())
print(df.info())

df['new_Date'] = pd.to_datetime(df['Date'])

print(df.head())
print(df.info())
print(type(df['new_Date'][0]))

df.set_index('new_Date', inplace = True) # 인덱스 지정
df.drop('Date', axis=1, inplace = True) # 기존 열 삭제

print(df.head())
print(df.info())