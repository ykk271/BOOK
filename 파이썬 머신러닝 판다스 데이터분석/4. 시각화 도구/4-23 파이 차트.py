import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')

df = pd.read_csv('C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part4/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleraetion', 'model year', 'origin', 'name']

# 데이터 개수 카운트
df['count'] = 1
df_origin = df.groupby('origin').sum() # origin 열을 기준으로 그룹화, 합계 연산
print(df_origin.head())

df_origin.index = ['USA', 'EU', 'JPN']

df_origin['count'].plot(kind='pie',
                        figsize=(7, 5),
                        autopct='%1.1f%%', # 퍼센트 % 표시
                        startangle=10, # 파이 조각을 나누는 시작점(각도 표시)
                        colors=['chocolate', 'bisque', 'cadetblue']
                        )

plt.title('Model Origin', size=20)
plt.axis('equal')
plt.legend(labels=df_origin.index, loc='upper right') # 범례표시
plt.show()