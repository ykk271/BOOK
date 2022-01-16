import pandas as pd

dict_data = {'c0': [1,2,3], 'c1': [4,5,6], 'c2': [7,8,9], 'c3': [10,11,12]}

df = pd.DataFrame(dict_data)

print(type(df))
print(df)