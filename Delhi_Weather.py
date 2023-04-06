import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy.linalg as LA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.append('E:\\MS\\Sem2\\TimeSeriesAnalysis\\Toolbox')
from toolbox import *

data = pd.read_csv('E:\\datasets\\Delhi weather\\Testset.csv')

# ---- Preprocessing

print(data.columns)

# renaming columns
data.rename(columns={'datetime_utc': 'Datetime', ' _conds': 'conditions', ' _dewptm': 'dewpoint',
                     ' _fog': 'fog', ' _hail': 'hail', ' _heatindexm': 'heatindex', ' _hum': 'humidity',
                     ' _precipm': 'precipitation', ' _pressurem': 'pressure', ' _rain': 'rain', ' _snow': 'snow',
                     ' _tempm': 'temp', ' _thunder': 'thunder', ' _tornado': 'tornado', ' _vism': 'visibility',
                     ' _wdird': 'wdirdegrees', ' _wdire': 'winddirection', ' _wgustm': 'windgust',
                     ' _windchillm': 'windchill', ' _wspdm': 'windspeed'}, inplace=True)

# wind gust is the sudden increase in speed of wind
# vis is visibility (in km or miles)
# dewpoint can be in F or C
# snow can be in inches or cm
# temp can be in F or C
# windspeed and windgust can be in mph or kph
# wdird is wind direction in degrees
# pressure is in millibars (mb)
# humidity is in percentage

# since there are very less NON-NULL values in precipitation, windchill, windgust, heatindex
# the column cannot be imputed, so dropping that.

data.drop(columns=['precipitation', 'windchill', 'heatindex'
    , 'windgust'], inplace=True)

print(data.columns)
print(data.info())

# there are few missing values in 'temp', 'visibility', 'wdird', 'winddirection', and 'windspeed'
# but there are enough values, so will try to impute them and see their use-cases

data['Datetime'] = pd.to_datetime(data['Datetime'])

data.index = data['Datetime']
data = data.drop(columns=['Datetime'])

df = data.resample('D').mean()

df2 = data.groupby(data.index.date).agg({'temp': ['min', 'max']})
# reference chatgpt

df['minTemp'] = df2[('temp', 'min')]
df['maxTemp'] = df2[('temp', 'max')]

df = df.ffill()  # some datapoints (particular date) were missing as there were no entry for that date.
# this is the forward fill method which can fill the missing data points with the last entry.
# I used this forward fill method as temperature cannot vary too much over the few days (specially in New Delhi).
# Hence, did this for a better model.
# reference chatgpt

# plotting seaborn heatmap
fig = plt.figure(figsize=(16, 10))
ax = sns.heatmap(df.corr(), annot=True)
plt.title('Heatmap of Delhi Temperature Data')
plt.show()
# weird heatmap coming. Need to check
# reference: https://pythonbasics.org/seaborn-heatmap/

# plotting the time series for minTemp, maxTemp, and humidity
print(df.head().to_string())
df['minTemp'].plot(color='r', label='MinTemp')
df['maxTemp'].plot(color='b', label='MaxTemp')
plt.legend()
plt.show()

df['humidity'].plot(color='b', label='humidity')
plt.legend()
plt.show()

# Just from looking at the plot, the data looks stationary.
# We will plot the rolling mean and variance plot to know more about it.
# And finally we will do the statistical testing to confirm our findings.

# Rolling mean and variance plots
# Need to uncomment this part when I am done. Computationally expensive part
# rolling_mean_var(df['minTemp'], 'MinTemp')
# rolling_mean_var(df['maxTemp'], 'MinTemp')

# Rolling mean and variance plots suggest that the time series is stationary.
# But we need to confirm it using statistical testing.

# Statistical tests
print('ADF Test')
ADF_Cal(df['minTemp'])

print('KPSS Test')
kpss_test(df['minTemp'])

# from the statistical tests, we confirm that our data is stationary.


y = cal_autocorr(df['temp'], 800, 'mean Temperature')
# ymin = cal_autocorr(df['minTemp'], 50, 'Min Temperature')
# ymax = cal_autocorr(df['maxTemp'], 50, 'Max Temperature')
# ACF plot came as expected. Generally the temperature will depend on the past values too much
# And ACF will decay as the number of lags are increased.

m = df.resample('M').mean()

Temp = pd.Series(df['minTemp'].values, index=df.index, name='Min temperature')

STL = STL(Temp, period=365)
res = STL.fit()
fig = res.plot()
plt.suptitle('Trend, seasonality, and remainder plot of Temperature')
plt.xlabel('Time (in years)')
plt.tight_layout()
plt.show()

T = res.trend
S = res.seasonal
R = res.resid

Ft = max(0, 1 - np.var(R)/(np.var(T+R)))
print("Strength of Trend for this dataset is ", Ft)


Fs = max(0, 1 - np.var(R)/(np.var(S+R)))
print("Strength of seasonality for this dataset is ", Fs)
# Strength of Trend for this dataset is  0.08139067256765353
# Strength of seasonality for this dataset is  0.9398796696924627

# as expected, seasonality is very high, and trend is very low when period is set to 365

# splitting data into test and train
X = df.drop(['temp', 'minTemp', 'maxTemp'], axis=1)
y = df['temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# Holt Winter Method
# this model is just considering 'temp'
# we will consider 'minTemp', 'maxTemp', and 'humidity' later
model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=365).fit()

forecast = model.forecast(steps=len(y_test))

title = 'Winter-Holt forecasting'
plot_forecasting_models(y_train, y_test, forecast, title)

# Calculating MSE for Winter-Holt method
_, _, mse = cal_error_MSE(y_test, forecast)
print('MSE for Winter-Holt method:', mse)

# Now will forecast using base models and plot them in a subplot

# Average forecasting
_, forecast_average = average_forecasting(y_train, y_test)
_, _, mse_average = cal_error_MSE(y_test, forecast_average)
print('MSE for Average forecasting:', mse_average)

# Naive forecasting
_, forecast_Naive = Naive_forecasting(y_train, y_test)
_, _, mse_Naive = cal_error_MSE(y_test, forecast_Naive)
print('MSE for Naive forecasting:', mse_Naive)

# Drift forecasting
_, forecast_Drift = drift_forecasting(y_train, y_test)
_, _, mse_Drift = cal_error_MSE(y_test, forecast_Drift)
print('MSE for Drift forecasting:', mse_Drift)

# Simple Exponential Smoothing
L0 = y_train[0]
_, forecast_SES = ses(y_train, y_test, L0, alpha=0.5)
_, _, mse_SES = cal_error_MSE(y_test, forecast_SES)
print('MSE for SES forecasting:', mse_SES)

# subplots of the base models
fig = plt.figure(figsize=(16, 8))
axs = fig.subplots(2, 2)
plot_forecasting_models(y_train, y_test, forecast_average, 'Average Forecasting', axs=axs[0][0])
plot_forecasting_models(y_train, y_test, forecast_Naive, 'Naive Forecasting', axs=axs[0][1])
plot_forecasting_models(y_train, y_test, forecast_Drift, 'Drift Forecasting', axs=axs[1][0])
plot_forecasting_models(y_train, y_test, forecast_SES, 'SES Forecasting', axs=axs[1][1])
plt.tight_layout()
plt.show()
# MSE for Winter-Holt method: 15.66
# MSE for Average forecasting: 51.73
# MSE for Naive forecasting: 63.07
# MSE for Drift forecasting: 70.36
# MSE for SES forecasting: 51.11

# Lowest MSE till now is of Winter-Holt seasonal method

# Scaling the data using standard Scaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

condition_number = LA.cond(X_train_scaled)
print('condition_number:', condition_number)
# From the condition number we can see that there is not much collinearity in our dataset

X_svd = X_train.to_numpy()
H = np.matmul(X_svd.T, X_svd)

s, d, v = LA.svd(H)
print('SingularValues =', d)
# SingularValues = [6.04547993e+13 2.70315928e+08 6.72024236e+06 3.02061569e+05
#  1.55522232e+05 2.52494063e+04 1.05676115e+02 7.94014223e+01
#  1.54329575e+01 1.27133958e-01 1.81119775e-02 4.14548803e-15]

# From the singular values, there are values which are very close to zero.
# So, it means that there is some collinearity in our dataset.




