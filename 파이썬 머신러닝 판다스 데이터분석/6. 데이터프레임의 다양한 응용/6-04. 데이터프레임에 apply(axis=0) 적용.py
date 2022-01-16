import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age','fare']]
print(df.head())

def min_max(x):
    return x.min() - x.max()

result = df.apply(min_max) # 기본값 axis = 0

print(result.head())
print(type(result))



