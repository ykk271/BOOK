import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'sex', 'class', 'fare' , 'survived']]

grouped = df.groupby(['class', 'sex'])

gdf = grouped.mean()
print(gdf)

print(gdf.loc['First'])

print(gdf.loc[('First', 'female')])

# sex 값이 male인 행을 선택하여 선택
print(gdf.xs('male', level='sex'))



