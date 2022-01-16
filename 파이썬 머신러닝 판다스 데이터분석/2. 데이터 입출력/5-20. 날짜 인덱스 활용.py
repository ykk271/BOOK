import pandas as pd

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'stock-data.csv')

df['new_Date'] = pd.to_datetime(df['Date'])
df.set_index('new_Date', inplace = True)

print(df.head())
print(df.index)

df_y = df['2018']
print(df_y)

df_ym = df.loc['2018-07']
print(df_ym)

df_ym_cols = df.loc['2018-07', 'Start':'High']
print(df_ym_cols)

df_ymd = df['2018-07-02']
print(df_ymd)

df_ymd_range = df['2018-06-25':'2018-06-20']
print(df_ymd_range)

today = pd.to_datetime('2018-12-25')
df['time_delta'] = today - df.index
df.set_index('time_delta', inplace = True)
print(df.head())
df_180 = df['180 days':'189 days']
print(df_180)


