import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

# 스테일 테마 설정 5가지: darkgrid, whitegrid, dark, white, ticks
sns.set_style('whitegrid')

# 피벗테이블 범주형 변수를 각각 행, 열로 재구분하여 정의
# aggfunc = 'size' 옵션은 데이터 값의 크기를 기준으로 집계
table = titanic.pivot_table(index=['sex'], columns=['class'], aggfunc='size')

sns.heatmap(table,
            annot=True, fmt='d', # 데이터 값 표시 여부, 정수형 포맷
            cmap='YlGnBu', # 컬러맵
            linewidth=.5,
            cbar=False)

plt.show()