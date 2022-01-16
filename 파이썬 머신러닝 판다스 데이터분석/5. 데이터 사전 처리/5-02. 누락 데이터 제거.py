import seaborn as sns

df = sns.load_dataset('titanic')

missing_df = df.isnull()

for col in missing_df.columns:
    missing_count = missing_df[col].value_counts() # 각 열의 NaN 개수 파악

    try:
        print(col, ': ', missing_count[True]) # NaN이 있으면 개수 출력
    except:
        print(col, ': ', 0) # NaN 값이 없으면 0개 출력

print(missing_count)

print(missing_df['embark_town'].value_counts())

# NaN값이 500개 이상인 열을 모두 제거
df_thresh = df.dropna(axis=1, thresh=500)
print(df_thresh.columns)

# age 열에 나이 데이터가 없는 모든 행 제거
df_age = df.dropna(subset=['age'], how ='any', axis=0) # how='all' 하나라도 NaN이면 제거
print(len(df_age))


