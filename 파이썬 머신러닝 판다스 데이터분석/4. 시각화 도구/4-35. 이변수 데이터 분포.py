import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

sns.set_style('whitegrid')

titanic_pair = titanic[['age', 'pclass', 'fare']]

g = sns.pairplot(titanic_pair)

plt.show()
