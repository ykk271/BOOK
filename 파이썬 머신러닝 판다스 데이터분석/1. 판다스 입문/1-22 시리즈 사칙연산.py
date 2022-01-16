import pandas as pd

student1 = pd.Series({'국어':100, '영어':80, '수학':90})
student2 = pd.Series({'수학':10, '국어':20, '영어':30})

print(student1)
print(student2)

add = student1 + student2
sub = student1 - student2
multi = student1 * student2
div = student1 / student2

print(type(div))
result = pd.DataFrame([add, sub, multi, div], index = {'+','-','x','/'})

print(result)