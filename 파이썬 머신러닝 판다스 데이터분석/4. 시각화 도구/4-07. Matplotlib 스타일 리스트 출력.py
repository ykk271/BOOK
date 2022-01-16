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

sr_one = df_seoul.loc['경기도']

plt.style.use('ggplot')

plt.figure(figsize = (14, 5)) # 인치 단위

plt.xticks(rotation='vertical') # x 눈금 라벨 회전하기

plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)

plt.title('서울 -> 경기 인구 이동')
plt.xlabel('기간')
plt.ylabel('이동 인구수')

plt.legend(labels=['서울 -> 경기'], loc = 'best')

plt.ylim(50000, 800000)

plt.annotate('',
             xy=(20, 620000),
             xytext = (2, 290000),
             xycoords='data',
             arrowprops=dict(arrowstyle='->', color='skyblue', lw=5),
             )

plt.annotate('',
             xy=(47, 450000),
             xytext = (30, 580000),
             xycoords='data',
             arrowprops=dict(arrowstyle='->', color='skyblue', lw=5),
             )

plt.annotate('인구수 이동 증가(1970-1995)',
             xy=(10, 400000),
             rotation=25,
             va='baseline',
             ha='center',
             fontsize=15,
             )

plt.annotate('인구수 이동 감소(1955-2017)',
             xy=(40, 500000),
             rotation=-11,
             va='baseline',
             ha='center',
             fontsize=15,
             )

plt.show()

