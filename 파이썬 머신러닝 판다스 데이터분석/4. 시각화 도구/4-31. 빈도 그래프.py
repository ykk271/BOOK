import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

sns.set_style('whitegrid')

titanic.head()

fig=plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

sns.countplot(x='class', palette='Set1', data=titanic, ax=ax1)

sns.countplot(x='class', hue='who' ,palette='Set2', data=titanic, ax=ax2)

sns.countplot(x='class', hue='who' ,palette='Set3', dodge=False, data=titanic, ax=ax3) # 누적

ax1.set_title('titanic class')
ax2.set_title('titanic class - who')
ax3.set_title('titanic class - who(stacked)')

plt.show()


