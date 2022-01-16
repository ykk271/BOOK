import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part3/auto-mpg.csv')

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleration', 'moder year', 'origin', 'name']

df.plot(x = 'weight', y = 'mpg', kind = 'scatter')

plt.show()