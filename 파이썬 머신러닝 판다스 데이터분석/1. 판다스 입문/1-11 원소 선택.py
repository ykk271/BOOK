import pandas as pd

exam_data = {'이름': [ '서준', '우현', '인아'],
             '수학': [100, 90, 80],
             '영어': [98, 89, 99],
             '음악': [99, 12, 11],
             '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data)

df.set_index('이름', inplace = True)
print(df)

a = df.loc['서준', '음악']
print(a)
b = df.iloc[0,2]
print(b)

c = df.loc['서준',['음악', '체육']]
print(c)
d = df.iloc[0, [2, 3]]
print(d)
e = df.loc['서준', '음악':'체육']
print(e)
f = df.iloc[0, 2:]
print(f)
g = df.loc[['서준', '우현'], ['음악', '체육']]
print(g)
h = df.iloc[[0, 1],[2, 3]]
print(h)
i = df.loc['서준':'우현', '음악':'체육']
print(i)
j = df.iloc[0:3, 2:]
print(j)