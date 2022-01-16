import pandas as pd

ts_ms = pd.date_range(start='2019-01-01',
                      end=None,
                      periods=6,
                      freq='MS', # 시간간격: 월의 시작일
                      tz='Asia/Seoul')

print(ts_ms)

ts_me = pd.date_range('2019-01-01',
                      periods=6,
                      freq='M', # 월의 마지막
                      tz='Asia/Seoul')
print(ts_me)

ts_3m = pd.date_range('2019-01-01',
                      periods=6,
                      freq='3M', # 월의 마지막
                      tz='Asia/Seoul')

print(ts_3m)


