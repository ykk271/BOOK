import pandas as pd

exam_data = {'이름': [ '서준', '우현', '인아'],
             '수학': [100, 90, 80],
             '영어': [98, 89, 99],
             '음악': [99, 12, 11],
             '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data)

df.loc[3] = 0
print(df)
df.loc[4] = ['동규', 95, 10, 22, 33]
print(df)

df.loc['행5'] = df.loc[3]
print(df)
