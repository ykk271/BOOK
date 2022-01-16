import pandas as pd

df = pd.DataFrame({'c1': ['a','b','a','c','b'],
                   'c2': [1, 1, 1, 1, 1],
                   'c3': [1, 1, 2, 3, 2]})

print(df)

# 데이터프레임 전체 행 데이터 중에서 중복값 찾기
df_dup = df.duplicated()

print(df_dup)

col_dup = df['c3'].duplicated()

print(col_dup)



