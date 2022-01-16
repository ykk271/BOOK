import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age','fare']]
print(df.head())

def missing_value(x):
    return x.isnull()

def missing_count(x):
    return missing_value(x).sum()

def total_number_missing(x):
    return missing_count(x).sum()

result_df = df.pipe(missing_value)

print(df.head())
print(result_df.head())
print(type(result_df))

result_series = df.pipe(missing_count)
print(result_series)
print(type(result_series))

result_value = df.pipe(total_number_missing)
print(result_value)
print(type(result_value))

