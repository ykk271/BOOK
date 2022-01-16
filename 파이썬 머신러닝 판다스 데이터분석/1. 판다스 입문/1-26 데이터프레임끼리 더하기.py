import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]
print(df.tail(5))
print(type(df))

addition = df + 10
print(addition.tail())

sub = addition - df
print(sub)