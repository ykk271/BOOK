import pandas as pd
import numpy as np

student1 = pd.Series({'국어':np.nan, '영어': 80, '수학': 90})
student2 = pd.Series({'수학':80, '국어':90})

print(student1)
print(student2)

add = student1 + student2
sub = student1 - student2
multi = student1 * student2
div = student1 / student2

print(type(div))
result = pd.DataFrame([add, sub, multi, div], index = {'+','-','x','/'})

print(result)

