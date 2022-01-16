import seaborn as sns

df = sns.load_dataset('titanic')

print(df['age'].head(10))

mean_age = df['age'].mean(axis=0) # NaN값 제외 평균 계산
df['age'].fillna(mean_age,inplace=True)

print(mean_age)
print(df['age'].head(10))



