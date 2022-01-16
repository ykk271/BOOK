import pandas as pd

exam_data = {'이름': [ '서준', '우현', '인아'],
             '수학': [100, 90, 80],
             '영어': [98, 89, 99],
             '음악': [99, 12, 11],
             '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data)

print(df)

df = df.transpose()
print(df)

df = df.T
print(df)