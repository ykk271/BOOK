import pandas as pd

tup_data = ('예광', '2010-05-10', '남', True)
sr = pd.Series(tup_data, index = ['이름', '날짜', '성별', '장애여부'])

print(sr)

print(sr[0])
print(sr['이름'])

print(sr[[1, 2]])
print(sr[['날짜', '성별']])


