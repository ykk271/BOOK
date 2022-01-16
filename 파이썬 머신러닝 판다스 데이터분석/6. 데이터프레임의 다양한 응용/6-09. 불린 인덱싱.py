import seaborn as sns

titanic = sns.load_dataset('titanic')

mask1 = (titanic.age >= 10) & (titanic.age < 20)
df_teenage = titanic.loc[mask1, :]
print(df_teenage.head())

mask2 = (titanic.age < 10) & (titanic.sex == 'female')
df_girl = titanic.loc[mask2, :]
print(df_girl.head())

mask3 = (titanic.age < 10) | (titanic.age >= 60)
df_weak = titanic.loc[mask3, :]
print(df_weak.head())
