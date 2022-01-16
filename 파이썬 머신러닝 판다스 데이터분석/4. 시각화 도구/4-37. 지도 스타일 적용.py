import folium

seoul_map2 = folium.Map(location=[37.55,126.98], zoom_start=12, tiles = 'Stamen Terrain')
seoul_map3 = folium.Map(location=[37.55,126.98], zoom_start=12, tiles = 'Stamen Toner')

seoul_map2.save('./seoul2.html')
seoul_map3.save('./seoul3.html')



