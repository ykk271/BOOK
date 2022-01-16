import pandas as pd

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'stock-data.csv')

df['new_Date'] = pd.to_datetime(df['Date'])

print(df.head())

df['Year'] = df['new_Date'].dt.year
df['Month'] = df['new_Date'].dt.month
df['Day'] = df['new_Date'].dt.day

print(df.head())

# Timestamp를 Period로 변환하여 연-월-일 표기 변경
df['Date_yr'] = df['new_Date'].dt.to_period(freq='A')
df['Date_m'] = df['new_Date'].dt.to_period(freq='M')
print(df.head())

df.set_index('Date_m', inplace = True)
print(df.head())


