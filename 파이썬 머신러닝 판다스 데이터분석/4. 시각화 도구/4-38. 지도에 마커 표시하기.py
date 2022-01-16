import folium
import pandas as pd

df = pd.read_excel('C:/Users/rladp/OneDrive/바탕 화면/데이터분석/Pandas Data Analytics/source/part4/서울지역 대학교 위치.xlsx', engine='openpyxl', index_col=0)

df.head()

seoul_map = folium.Map(location=[37.55,126.98], zoom_start=12, tiles = 'Stamen Terrain')

for name, lat, lng in zip(df.index, df.위도, df.경도):
    folium.Marker([lat, lng], popup=name). add_to(seoul_map)
    print(name)

seoul_map.save('./seoul_colleges.html') 





