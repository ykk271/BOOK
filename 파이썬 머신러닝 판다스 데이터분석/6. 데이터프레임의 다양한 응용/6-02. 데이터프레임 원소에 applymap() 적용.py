import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age','fare']]

def add_10(n):
    return n + 10

df_map = df.applymap(add_10)
print(df_map.head())