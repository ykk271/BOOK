import pandas as pd

pr_m = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='M')

print(pr_m)

pr_h = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='H')

print(pr_h)

pr_2h = pd.period_range(start='2019-01-01',
                       end=None,
                       periods=3,
                       freq='2H')

print(pr_2h)


 