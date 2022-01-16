import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')
print(df.head())

pd.set_option('display.max_columns', 15)
print(df.head())

# NaN���� ���� �� ����
rdf = df.drop(['deck', 'embark_town'], axis=1)
print(rdf.columns.values)

# age ���� ���� �����Ͱ� ���� ��� �� ����
rdf = rdf.dropna(subset=['age'], how='any', axis=0)
print(len(rdf))

# embarked���� NaN ���� �¼����� �߿��� ���� ���� ������ ������ ġȯ
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()
print(most_freq)

print(rdf.describe(include='all'))

rdf['embarked'].fillna(most_freq, inplace=True)

ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
print(ndf.head())

onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)

one_hot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, one_hot_embarked], axis=1)
print(ndf.head())

x=ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male',
       'town_C', 'town_Q', 'town_S']]

y=ndf['survived']

# ������ ����ȭ
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.3, random_state=10)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

y_hat = knn.predict(x_test)

print(y_hat)
print(y_test)

from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)

knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report)