import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like 
import pandas_datareader.data as web
import quandl
import numpy as np


style.use('ggplot')

start = "2018-02-23"
end = "2018-05-23"


df = quandl.get("WIKI/TSLA", start_date=start, end_date=end)




#df.to_csv('TSLA.csv')

df['100ma'] = df['Adj. Close'].rolling(window=100,min_periods=0).mean()
print(df)
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj. Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()
