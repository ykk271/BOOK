import pandas as pd

exam_data = {'수학': [100, 90, 80], '영어': [98, 89, 99],
             '음악': [99, 12, 11], '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])

print(df)

df2 = df[:]
df2.drop('우현', inplace = True)
print(df2)
# inplace = 반환된 객체를 기존 변수에 저장하여 대체하는 방법

df3 = df[:]
df3.drop(['우현', '인아'], axis = 0, inplace = True)
print(df3)
# axis = 1로 하면 열이 삭제됨




