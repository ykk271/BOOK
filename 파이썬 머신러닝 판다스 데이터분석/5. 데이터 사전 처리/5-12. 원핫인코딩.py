import pandas as pd
import numpy as np

data_path = 'C:/Users/rladp/OneDrive/���� ȭ��/�����ͺм�/Pandas Data Analytics/source/part5/'

df = pd.read_csv(data_path+'auto-mpg.csv')

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

df['horsepower'].replace('?', np.nan, inplace=True)  
df.dropna(subset=['horsepower'], axis=0, inplace=True)
df['horsepower'] = df['horsepower'].astype('float')

print(df['horsepower'].sample(10))

# np.historam �Լ��� 3���� bin���� ������ ��谪 ����Ʈ ã��
count, bin_dividers = np.histogram(df['horsepower'], bins=3)

bin_names = ['�����', '�������', '�����']

df['hp_bin'] = pd.cut(x=df['horsepower'],
                      bins=bin_dividers,
                      labels=bin_names,
                      include_lowest=True)  # ù ��谪 ����


print(df[['horsepower', 'hp_bin']].head(15))

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

onehot_labeled = label_encoder.fit_transform(df['hp_bin'])
print(onehot_labeled)
print(type(onehot_labeled))

onehot_reshaped = onehot_labeled.reshape(len(onehot_labeled), 1)
print(onehot_reshaped)
print(type(onehot_reshaped))

# �����ķ� ��ȯ
onehot_fitted = onehot_encoder.fit_transform(onehot_reshaped)
print(onehot_fitted)
print(type(onehot_fitted))