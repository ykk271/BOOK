import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

# 스테일 테마 설정 5가지: darkgrid, whitegrid, dark, white, ticks
sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# 선형회귀선 표시
sns.regplot(x = 'age',
            y = 'fare',
            data=titanic,
            ax=ax1)

sns.regplot(x='age',
            y='fare',
            data=titanic,
            ax=ax2,
            fit_reg=False)

plt.show()