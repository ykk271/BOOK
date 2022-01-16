import pandas as pd

exam_data = {'이름': [ '서준', '우현', '인아'],
             '수학': [100, 90, 80],
             '영어': [98, 89, 99],
             '음악': [99, 12, 11],
             '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data)

print(df)

ndf = df.set_index(['이름'])
print(ndf)


ndf = df.set_index(['음악'])
print(ndf)


ndf = df.set_index(['이름', '수학'])
print(ndf)

