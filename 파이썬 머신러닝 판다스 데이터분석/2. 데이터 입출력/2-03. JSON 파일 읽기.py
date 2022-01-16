import pandas as pd

file_path = 'C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part2/'
file_name = 'read_json_sample.json'
file_fullName = file_path+file_name

df = pd.read_json(file_fullName)
print(df)
print(df.index)

