import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'sex', 'class', 'fare' , 'survived']]

grouped = df.groupby(['class'])

std_all = grouped.std()
print(std_all)
print(type(std_all))

std_fare = grouped.fare.std()
print(std_fare)
print(type(std_fare))

def min_max(x):
    return x.max() - x.min()

# 메소드 데이터 집계
agg_minmax = grouped.agg(min_max)
print(agg_minmax.head())

# 여러 함수를 각 열에 동일하게 적용하여 집계
agg_all = grouped.agg(['min', 'max'])
print(agg_all.head())

# 각 열마다 다른 함수를 적용하여 집계
agg_sep = grouped.agg({'fare': ['min', 'max'], 'age':'mean'})
print(agg_sep.head())

