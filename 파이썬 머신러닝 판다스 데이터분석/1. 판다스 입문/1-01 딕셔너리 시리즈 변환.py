import pandas as pd

# 딕셔너리 생성
dict_data = {'a': 1, 'b': 2, 'c' : 3}

# 시리즈로 변환
sr = pd.Series(dict_data)

print(type(sr))
print(sr)

