import pandas as pd
import folium

data_path = 'C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part7/'
df = pd.read_excel(data_path+'2016_middle_shcool_graduates_report.xlsx', engine='openpyxl')

pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.unicode.east_asian_width', True)

print(df.columns.values)

print(df.info())
print(df.describe())

# 지도에 위치 표시
mschool_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain',
                         zoom_start=12)

for name, lat, lng in zip(df.학교명, df.위도, df.경도):
    folium.CircleMarker([lat, lng],
                       radius=5,
                       color='brown',
                       fill=True,
                       fill_color='coral',
                       fill_opacity=0.7,
                       popup=name
     ).add_to(mschool_map)


mschool_map.save('./seoul_mschool_location.html')

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

onehot_location = label_encoder.fit_transform(df['지역'])
onehot_code = label_encoder.fit_transform(df['코드'])
onehot_type = label_encoder.fit_transform(df['유형'])
onehot_day = label_encoder.fit_transform(df['주야'])

df['location'] = onehot_location
df['code'] = onehot_code
df['type'] = onehot_type
df['day'] = onehot_day

print(df.head())

from sklearn import cluster

columns_list = [9, 10, 13]
x = df.iloc[:, columns_list]

print(x[:5])

x = preprocessing.StandardScaler().fit(x).transform(x)

dbm = cluster.DBSCAN(eps=0.2, min_samples=5)

dbm.fit(x)

cluster_label = dbm.labels_
print(cluster_label)

df['Cluster'] = cluster_label
print(df.head())

grouped_cols = [0, 1, 3] + columns_list
grouped = df.groupby('Cluster')

for key, group in grouped:
    print('* key: ', key)
    print('* number : ', len(group))
    print(group.iloc[:, grouped_cols].head())
    print('\n')

colors = {-1:'gray', 0:'coral', 1:'blue', 2:'green', 3:'red', 4:'purple',
          5:'orange', 6:'brown', 7:'brick', 8:'yellow', 9:'megenta', 10:'cyan'}

cluster_map = folium.Map(location = [37.55, 126.98], tiles='Stamen Terrain',
                         zoom_start=12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster):
    folium.CircleMarker([lat, lng],
                        radius = 5,
                        color = colors[clus],
                        full=True,
                        fill_color=colors[clus],
                        fill_opacity=0.7,
                        popup=name
                        ).add_to(cluster_map)

cluster_map.save('./seoul_mschool_cluster.html')

columns_list2 = [9, 10, 13, 22]
x2 = df.iloc[:, columns_list2]
print(x2[:5])

x2 = preprocessing.StandardScaler().fit(x2).transform(x2)
dbm2 = cluster.DBSCAN(eps=0.2, min_samples=5)
dbm2.fit(x2)
df['Cluster2'] = dbm2.labels_

grouped2_cols = [0, 1, 3] + columns_list2
grouped2 = df.groupby('Cluster2')

for key, group in grouped2:
    print('*key : ', key)
    print('* number: ', len(group))
    print(group.iloc[:, grouped2_cols].head())
    print('\n')

cluster2_map = folium.Map(onehot_location=[37.55,126.98], tiles='Stamen Terrain',
                          zoom_start=12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster2):  
    folium.CircleMarker([lat, lng],
                        radius=5,                   # 원의 반지름
                        color=colors[clus],         # 원의 둘레 색상
                        fill=True,
                        fill_color=colors[clus],    # 원을 채우는 색
                        fill_opacity=0.7,           # 투명도    
                        popup=name
    ).add_to(cluster2_map)

cluster2_map.save('./seoul_mschool_cluster2.html')


columns_list3 = [9,10]
x3 = df.iloc[:, columns_list3]
print(x3[:5])

x3 = preprocessing.StandardScaler().fit(x3).transform(x3)
dbm3 = cluster.DBSCAN(eps = 0.2, min_samples=5)
dbm3.fit(x3)
df['Cluster3'] = dbm3.labels_


grouped3_cols = [0, 1, 3] + columns_list3 
grouped3 = df.groupby('Cluster3')

for key, group in grouped3:
    print('* key :', key)
    print('* number :', len(group)
    print(group.iloc[:, grouped3_cols].head())
    print('\n)')



cluster3_map = folium.Map(onehot_location=[37.55,126.98], tiles='Stamen Terrain',
                          zoom_start=12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster3):
    folium.CircleMarker([lat, lng],
                         radius=5,
                         color=colors[clus],
                         fill=True,
                         fill_color=colors[clus],
                         fill_opacity=0.7,
                         popup=name
            ).add_to(cluster3_map)

cluster3_map.save('./seoul_mschool_cluster3.html')