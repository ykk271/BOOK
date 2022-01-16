import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

sns.set_style('whitegrid')

j1 = sns.jointplot(x='fare', y='age', data=titanic)

# 회귀선
j2 = sns.jointplot(x='fare', y='age', kind='reg', data=titanic)

# 육각그래프
j3 = sns.jointplot(x='fare', y='age', kind='hex', data=titanic)

# 커널 밀집 그래프
j4 = sns.jointplot(x='fare', y='age', kind='kde', data=titanic)


j1.fig.suptitle('titanic fare - scatter', size=15)
j2.fig.suptitle('titanic fare - reg', size=15)
j3.fig.suptitle('titanic fare - hex', size=15)
j4.fig.suptitle('titanic fare - kde', size=15)

plt.show()