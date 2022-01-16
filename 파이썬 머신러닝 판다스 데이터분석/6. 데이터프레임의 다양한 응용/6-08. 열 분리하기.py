import pandas as pd

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part6/'

df = pd.read_excel(data_path+'주가데이터.xlsx', engine='openpyxl')

print(df.head())
print(df.dtypes)
df['연월일'] = df['연월일'].astype('str')
dates = df['연월일'].str.split('-')
print(dates.head())

df['연'] = dates.str.get(0)
df['월'] = dates.str.get(1)
df['일'] = dates.str.get(2)

print(df.head())

