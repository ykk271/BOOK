import pandas as pd

file_path = 'C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part2/'
file_name = '남북한발전전력량.xlsx'
file_fullName = file_path+file_name
print(file_fullName)

# required pip install openpyxl
df1 = pd.read_excel(file_fullName, engine='openpyxl')
print(df1)

df2 = pd.read_excel(file_fullName, engine='openpyxl', header=None)
print(df2)

