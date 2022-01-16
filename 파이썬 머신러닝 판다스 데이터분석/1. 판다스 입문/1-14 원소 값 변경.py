import pandas as pd

exam_data = {'이름': [ '서준', '우현', '인아'],
             '수학': [100, 90, 80],
             '영어': [98, 89, 99],
             '음악': [99, 12, 11],
             '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data)

df.set_index('이름', inplace=True)
print(df)

df.iloc[0][3] = 999
print(df)

df.loc['서준', '음악'] = 998
print(df)

df.loc['서준', ['음악', '체육']] = 555
print(df)

df.loc['서준', ['음악', '체육']] = 111, 222
print(df)

