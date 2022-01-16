import seaborn as sns

df = sns.load_dataset('titanic')

print(df['embark_town'][825:830])

most_freq = df['embark_town'].value_counts(dropna=True).idxmax()
print(most_freq)

df['embark_town'].fillna(most_freq, inplace=True)

print(df['embark_town'][825:830])

'''
누락 데이터가 NaN으로 표시되지 않는 경우 
df.replace('?', np.nan, inplace = True)
'''