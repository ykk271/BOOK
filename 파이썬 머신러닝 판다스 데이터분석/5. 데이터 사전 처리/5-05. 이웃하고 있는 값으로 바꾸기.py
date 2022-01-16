import seaborn as sns

df = sns.load_dataset('titanic')

print(df['embark_town'][825:830])

# NaN을 바로 앞 행의 값으로 바꾸기
df['embark_town'].fillna(method='ffill', inplace = True)

print(df['embark_town'][825:830])

