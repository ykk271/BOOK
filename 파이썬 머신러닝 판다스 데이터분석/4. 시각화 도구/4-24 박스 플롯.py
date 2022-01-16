import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 오류 해결
from matplotlib import font_manager,rc
font_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part4/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

plt.style.use('seaborn-poster')
plt.rcParams['axes.unicode_minus']=False 

df = pd.read_csv('C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part4/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleraetion', 'model year', 'origin', 'name']

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.boxplot(x=[df[df['origin'] == 1]['mpg'],
               df[df['origin'] == 2]['mpg'],
               df[df['origin'] == 3]['mpg']],
            labels=['USA','EU','JAPAN'])

ax2.boxplot(x=[df[df['origin'] == 1]['mpg'],
               df[df['origin'] == 2]['mpg'],
               df[df['origin'] == 3]['mpg']],
            labels=['USA','EU','JAPAN'],
            vert=False)

ax1.set_title('제조국가별 연비 분포(수직 박스 플롯)')
ax2.set_title('제조국가별 연비 분포(수평 박스 플롯)')

plt.show()

'''
파이썬 그래프 갤러리
https://python-graph-gallery.com
'''