import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'sex', 'class', 'fare' , 'survived']]

grouped = df.groupby(['class'])

grouped_filter = grouped.filter(lambda x: len(x) >= 200)
print(grouped_filter.head())
print(type(grouped_filter))

grouped_filter = grouped.filter(lambda x: x.age.mean() < 30)
print(grouped_filter.tail())
print(type(grouped_filter))

agg_grouped = grouped.apply(lambda x: x.describe())
print(agg_grouped)

def z_xcore(x):
    return (x - x.mean())/x.std()

age_zscore = grouped.age.apply(z_score)
print(age_zscore.head())

age_filter = grouped.apply(lambda x: x.age.mean() < 30)
print(age_filter)

for x in age_filter.index:
    if age_filter[x] == True:
        age_filter_df = grouped.get_group(x)
        print(age_filter_df.head())
        print('\n')



