import pandas as pd
import numpy as np

uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path, header=None)

df.columns = ['id', 'clump', 'cell_size', 'cell_shape', 'adhesion', 'eptihial',
              'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses', 'class']

pd.set_option('display.max_columns', 15)


print(df.head())
print(df.info())
print(df.describe())

print(df['bare_nuclei'].unique())

df['bare_nuclei'].replace('?', np.nan, inplace=True)
df.dropna(subset=['bare_nuclei'], axis=0, inplace=True)
df['bare_nuclei'] = df['bare_nuclei'].astype('int')

print(df.describe())


x = df[['clump', 'cell_size', 'cell_shape', 'adhesion', 'eptihial',
        'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']]
y = df['class']

from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

print(x_train.shape)
print(x_test.shape)

from sklearn import tree
tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)

tree_model.fit(x_train, y_train)

y_hat = tree_model.predict(x_test)

print(y_hat[0:10])
print(y_test.values[0:10])

from sklearn import metrics
tree_matrix = metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)
tree_report = metrics.classification_report(y_test, y_hat)
print(tree_report)