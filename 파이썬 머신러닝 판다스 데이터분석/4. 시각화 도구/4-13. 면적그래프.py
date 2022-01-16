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


col_years = list(map(str, range(1970, 2018)))
df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도','전라남도'], col_years]
df_4 = df_4.transpose()
plt.style.use('ggplot')

df_4.index = df_4.index = df_4.index.map(int)

df_4.plot(kind='area', stacked=False, alpha=0.2, figsize=(20,10))

plt.title('서울 -> 타시도 인구 이동', size=30)
plt.ylabel('이동 인구 수', size=20)
plt.xlabel('기간', size=20)
plt.legend(loc='best', fontsize=15)

plt.show()
