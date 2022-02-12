import pandas as pd
import numpy as np

data_path = 'D:/KAGGLE DATA/santander-product-recommendation/'

train = pd.read_csv(data_path+'train_ver2.csv')
train.shape

train.head
train.columns

# for loop로 모든 변수의 첫 5줄 미리보기
for col in train.columns:
    print('{}\n'.format(train[col].head()))

train.info()

# 수치형 변수 살펴보기
num_cols = [col for col in train.columns[:24] if train[col].dtype in ['int64', 'float64']]
num_cols # 7개
for col in num_cols:
    print('{}\n'.format(train[col].describe()))

# 범주형 변수 살펴보기
cat_cols = [col for col in train.columns[:24] if train[col].dtype in ['object']]
cat_cols
print(train[cat_cols].describe())

# 범수형 변수의 고유값을 직접 출력해보기
for col in cat_cols:
    uniq = np.unique(train[col].astype(str))
    print('-'*50)
    print('# col {}, n_uniq {}, uniq: {}'.format(col, len(uniq), uniq))

# 시각화로 데이터 살펴보기
import matplot
import matplotlib.pyplot as plt
import seaborn as sns

skip_cols = ['ncodpers', 'renta'] # 고유값이 너무 많음
for col in train.columns:
    if col in skip_cols:
        continue

    f, ax = plt.subplots(figsize=(20,15))
    sns.countplot(x=col, data=train, alpha = 0.5)
    plt.show()


