import pandas as pd

file_path = 'C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part2/'
file_name = 'read_csv_sample.csv'
file_fullName = file_path+file_name

df1 = pd.read_csv(file_fullName)
print(df1)

df2 = pd.read_csv(file_fullName, header=None)
print(df2)

df3 = pd.read_csv(file_fullName, index_col=None)
print(df3)

df4 = pd.read_csv(file_fullName, index_col = 'c0')
print(df4)
