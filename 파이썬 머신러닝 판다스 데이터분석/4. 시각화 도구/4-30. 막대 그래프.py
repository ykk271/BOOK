import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

sns.set_style('whitegrid')

fig=plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

sns.barplot(x='sex',y='survived', data=titanic, ax=ax1)

sns.barplot(x='sex',y='survived',hue='class',data=titanic, ax=ax2)

sns.barplot(x='sex',y='survived',hue='class',dodge=False, data=titanic, ax=ax3)

ax1.set_title('titanic survived - sex')
ax1.set_title('titanic survived - sex/class')
ax1.set_title('titanic survived - sex/class(stacked)')

plt.show()