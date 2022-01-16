import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[0:4, 'survived':'age']

print(df)

columns = list(df.columns.values)

print(columns)

columns_sorted = sorted(columns)
df_sorted = df[columns_sorted]
print(df_sorted)

columns_reversed = reversed(columns)
df_reversed = df[columns_reversed]
print(df_reversed)

columns_customed = ['pclass', 'sex', 'age', 'survived']
df_customed = df[columns_customed]
print(df_customed)



