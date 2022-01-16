#########
# 왜 안되는지 이유를 모르겠음


import pandas as pd


file_path = 'C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part2/'
file_name = './sample.html'
file_fullName = file_path+file_name

tables = pd.read_html(file_fullName)

