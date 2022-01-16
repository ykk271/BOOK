import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')

df = pd.read_csv('C:/Users/rladp/OneDrive/바탕 화면/판다스 데이터분석/DataAnalytics/source/part4/auto-mpg.csv', header=None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
              'acceleraetion', 'model year', 'origin', 'name']

cylinders_size = df.cylinders/df.cylinders.max() * 300

df.plot(kind='scatter', x='weight', y='mpg', c='coral', s=cylinders_size, figsize=(10,5), alpha = 0.3)
plt.title('Scatter Plot - mpg vs weight')
plt.show()
