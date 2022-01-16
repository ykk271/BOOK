import pandas as pd

df = pd.DataFrame({'c1': ['a','a','b','a','b'],
                   'c2': [1, 1, 1, 1, 1],
                   'c3': [1, 1, 2, 3, 2]})

print(df)

df2 = df.drop_duplicates()
print(df2) 

df3 = df.drop_duplicates(subset=['c2','c3'])
print(df3)

df4 = df.drop_duplicates(subset=['c1'])
print(df4)


