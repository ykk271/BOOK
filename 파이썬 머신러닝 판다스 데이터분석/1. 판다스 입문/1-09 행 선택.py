import pandas as pd

exam_data = {'수학': [100, 90, 80], '영어': [98, 89, 99],
             '음악': [99, 12, 11], '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data, index = ['서준','우현','인아'])

print(df)

label1 = df.loc['서준']
position1 = df.iloc[0]
print(label1)
print(position1)

label2 = df.loc[['서준', '우현']]
position2 = df.iloc[[0, 1]]
print(label2)
print(position2)

label3 = df.loc['서준':'우현']
position3 = df.iloc[0:1]
print(label3)
print(position3)

