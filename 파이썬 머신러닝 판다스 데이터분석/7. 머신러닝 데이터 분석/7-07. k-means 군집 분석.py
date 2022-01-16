import pandas as pd
import matplotlib.pyplot as plt

# Wholesale customers 데이터셋 가져오기 (출처: UCI ML Repository)
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path, header=0)

print(df.head())
print(df.info())
print(df.describe())

x = df.iloc[:,:]
print(x[:5])

from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
print(x[:1])

from sklearn import cluster

kmeans = cluster.KMeans(init = 'k-means++', n_clusters=5, n_init=10)
kmeans.fit(x)

cluster_label = kmeans.labels_
print(cluster_label)

df['Cluster'] = cluster_label
print(df.head())

fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

df.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1',
        colorbar=True, ax=ax1)

df.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set2',
        colorbar=True, ax=ax2)

plt.show()


mask = (df['Cluster'] == 0) | (df['Cluster'] == 4)
ndf = df[~mask]

fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ndf.plot(kind='scatter', x='Grocery', y='Frozen', c='Cluster', cmap='Set1',
        colorbar=True, ax=ax1)

ndf.plot(kind='scatter', x='Milk', y='Delicassen', c='Cluster', cmap='Set2',
        colorbar=True, ax=ax2)

plt.show()