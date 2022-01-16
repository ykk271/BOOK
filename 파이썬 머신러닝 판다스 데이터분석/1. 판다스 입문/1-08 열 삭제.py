import pandas as pd

exam_data = {'수학': [100, 90, 80], '영어': [98, 89, 99],
             '음악': [99, 12, 11], '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data, index = ['서준','우현','인아'])

print(df)

df4 = df[:]
df4.drop('수학', axis = 1, inplace = True)
print(df4)

df5 = df[:]
df5.drop(['영어', '음악'], axis = 1, inplace = True)
print(df5)

