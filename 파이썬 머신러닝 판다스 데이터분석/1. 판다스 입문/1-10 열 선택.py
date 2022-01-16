import pandas as pd

exam_data = {'이름': [ '서준', '우현', '인아'],
             '수학': [100, 90, 80],
             '영어': [98, 89, 99],
             '음악': [99, 12, 11],
             '체육': [100, 21, 11]}

df = pd.DataFrame(exam_data)

print(df)

print(type(df))

math1 = df['수학']
print(math1)
print(type(math1))

english = df.영어
print(english)
print(type(english))

music_gym = df[['음악','체육']]
print(music_gym)
print(type(music_gym))

T1 = df.iloc[::2]
print(T1)

T2 = df.iloc[0:3:2]
print(T2)

T3 = df.iloc[::-1]
print(T3)

# df.loc : 행
# df.iloc: 열

