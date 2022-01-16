import pandas as pd
import matplotlib.pyplot as plt

dataName = 'C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part4/시도별 전출입 인구수.xlsx'

df = pd.read_excel(dataName, engine='openpyxl',  header=0) # fillna 안됨
df.head()
df.tail()
df.isnull()
df = df.fillna(0)
df.tail()

df2 = pd.read_excel(dataName, engine='openpyxl',  header=0) 
df2 = df2.fillna(method='ffill') # 전 데이터로 결측값 채우기
df2.tail()

# 서울에서 다른 지역으로 이동한 데이터만 추출하여 정리 => 파트 6에서 자세히 다룰 예정
mask = (df2['전출지별'] == '서울특별시') & (df2['전입지별'] != '서울특별시')
df_seoul = df2[mask]
df_seoul = df_seoul.drop(['전출지별'], axis = 1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace = True)
df_seoul.set_index('전입지', inplace = True)

df_seoul.head(8)

# 서울에서 경기도로 이동한 인구 데이터 값만 선택
sr_one = df_seoul.loc['경기도']
sr_one.head()

plt.plot(sr_one.index, sr_one.values)
plt.show()

plt.plot(sr_one)
plt.show()


sr_one = df_seoul.loc['경기도']