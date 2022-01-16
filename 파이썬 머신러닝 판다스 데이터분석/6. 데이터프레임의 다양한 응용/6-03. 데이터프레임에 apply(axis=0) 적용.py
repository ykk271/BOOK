import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age','fare']]
print(df.head())

def missing_value(series):
    return series.isnull()

result = df.apply(missing_value, axis=0)

print(result.head())
print(type(result))
