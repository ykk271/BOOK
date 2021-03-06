import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age','fare']]
print(df.head())

df['ten'] = 10

print(df.head())

def add_two_obj(a, b):
    return a + b

df['add'] = df.apply(lambda x: add_two_obj(x['age'], x['ten']), axis = 1)

print(df.head())