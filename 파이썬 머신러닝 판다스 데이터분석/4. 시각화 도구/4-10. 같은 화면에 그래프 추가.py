import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager,rc

dataName = 'C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part4/시도별 전출입 인구수.xlsx'

font_path = 'C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part4/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df = pd.read_excel(dataName, engine='openpyxl',  header=0) 
df = df.fillna(method='ffill') # 전 데이터로 결측값 채우기
df.tail()

# 서울에서 다른 지역으로 이동한 데이터만 추출하여 정리 => 파트 6에서 자세히 다룰 예정
mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
df_seoul = df[mask]
df_seoul = df_seoul.drop(['전출지별'], axis = 1)
df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace = True)
df_seoul.set_index('전입지', inplace = True)

# 서울에서 경기도로 이동한 인구 데이터 값만 선택
sr_one = df_seoul.loc['경기도']

col_years = list(map(str, range(1970, 2018)))
df_3 = df_seoul.loc[['충청남도', '경상북도', '강원도'], col_years]

plt.style.use('ggplot')

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1, 1, 1)

# axe 객체에 plot 함수로 그래프 출력
ax.plot(col_years, df_3.loc['충청남도',:], marker='o', markerfacecolor='green',
        markersize=10, color = 'olive', linewidth=2, label = '서울 -> 충남')
ax.plot(col_years, df_3.loc['경상북도',:], marker='o', markerfacecolor='blue',
        markersize=10, color = 'skyblue', linewidth=2, label = '서울 -> 경북')
ax.plot(col_years, df_3.loc['강원도',:], marker='o', markerfacecolor='red',
        markersize=10, color = 'magenta', linewidth=2, label = '서울 -> 강원')

ax.legend(loc='best') # 최적화 위치

ax.set_title('서울 -> 충남, 경북, 강원 인구 이동', size = 20)

ax.set_xlabel('기간', size = 12)
ax.set_ylabel('이동 인구수', size = 12)

ax.set_xticklabels(col_years, rotation = 10)

ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)


plt.show()