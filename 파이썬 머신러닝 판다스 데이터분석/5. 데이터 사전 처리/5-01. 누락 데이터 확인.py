import seaborn as sns

df = sns.load_dataset('titanic')

df.head(10)

# deck 열의 NaN 개수 계산하기
nan_deck = df['deck'].value_counts(dropna=False)

print(nan_deck)

'''
누락 데이터를 직접 찾는 방법
isnull()
notnull()
'''

print(df.head().isnull())
print(df.head().notnull())

print(df.head().isnull().sum(axis=0))

