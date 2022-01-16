import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.unicode.east_asian_width', True)

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part6/'

df1 = pd.read_excel(data_path+'stock price.xlsx', engine='openpyxl', index_col = 'id')
df2 = pd.read_excel(data_path+'stock valuation.xlsx', engine='openpyxl', index_col = 'id')

df3 = df1.join(df2)
print(df3)

df4 = df1.join(df2, how='inner')
print(df4)

