import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics\/source/part3/남북한발전전력량.xlsx', engine='openpyxl')

df_ns = df.iloc[[0, 5], 3:]
df_ns.index = ['South', 'North']
df_ns.columns = df_ns.columns.map(int)
print(df_ns.head())

df_ns.plot()
plt.show()

tdf_ns = df_ns.T
print(tdf_ns.head())

tdf_ns.plot()
plt.show()

