import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

style.use('ggplot')

start = "2018-02-23"
end = "2018-05-23"


df = quandl.get("WIKI/TSLA")


forecast_out = int(30)
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X)

print(df)
print(df.tail())

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)

confidence = clf.score(X_test, y_test)
print(confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)

#df.to_csv('TSLA.csv')