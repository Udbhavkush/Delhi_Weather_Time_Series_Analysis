import math

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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from statsmodels.stats.outliers_influence import variance_inflation_factor


from tensorflow.keras import Sequential
# from keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dense, LSTM, Dropout  # ,CuDNNLSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam

sys.path.append('C:\\Users\\LENOVO\\PycharmProjects\\TimeSeriesLabs')
from toolbox import *

def forecast(y, T, h):
    T = T - 1
    y_hat = []
    for i in range(1, h+1):
        if i == 1:
            y_hat.append(0.7428 * y[T] + 0.0568 * y[T - 4] + 0.0428 * y[T - 6])
        elif i == 2:
            y_hat.append(0.7428 * y_hat[0] + 0.0568 * y[T - 3] + 0.0428 * y[T - 5])
        elif i == 3:
            y_hat.append(0.7428 * y_hat[1] + 0.0568 * y[T - 2] + 0.0428 * y[T - 4])
        elif i == 4:
            y_hat.append(0.7428 * y_hat[2] + 0.0568 * y[T - 1] + 0.0428 * y[T - 3])
        elif i == 5:
            y_hat.append(0.7428 * y_hat[3] + 0.0568 * y[T] + 0.0428 * y[T - 2])
        elif i == 6:
            y_hat.append(0.7428 * y_hat[4] + 0.0568 * y_hat[0] + 0.0428 * y[T - 1])
        elif i == 7:
            y_hat.append(0.7428 * y_hat[5] + 0.0568 * y_hat[1] + 0.0428 * y[T])
        else:
            y_hat.append(0.7428 * y_hat[i-1-1] + 0.0568 * y_hat[i - 5 - 1] + 0.0428 * y[i - 7 - 1])
    y_hat = np.array(y_hat)
    return y_hat


# data = pd.read_csv('E:\\datasets\\Delhi weather\\Testset.csv')
data = pd.read_csv('testset.csv')
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

plt.plot(df['temp'], label='Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature in Degree Celsius')
plt.title('Temperature VS Date Plot')
plt.grid()
plt.legend()
plt.show()

# plotting seaborn heatmap
fig = plt.figure(figsize=(16, 10))
ax = sns.heatmap(df.corr(), annot=True)
plt.title('Heatmap of Delhi Temperature Data')
plt.show()
# # reference: https://pythonbasics.org/seaborn-heatmap/
#
# # plotting the time series for minTemp, maxTemp, and humidity
# print(df.head().to_string())
df['minTemp'].plot(color='r', label='MinTemp')
df['maxTemp'].plot(color='b', label='MaxTemp')
plt.ylabel('Temperature in Degree Celsius')
plt.xlabel('Date')
plt.title('MinTemp and MaxTemp vs Date')
plt.grid()
plt.legend()
plt.show()
#
# df['humidity'].plot(color='b', label='humidity')
# plt.legend()
# plt.show()
# from the plot, we see that some temperature readings are inaccurate.
# i.e. temperature is touching around 70 degrees Celsius. It is inaccurate.

# removing inaccurate values
# Wikipedia page suggests that highest ever temperature of Delhi is 48.4 degree Celsius.
# So anything above this is not accurate.
# Also, minimum temperature ever recording In Delhi is -2 degree Celsius. So anything below this is not accurate.

df = df[df['maxTemp'] < 50]  # taking 50 degrees C as the cutoff value
df = df[df['minTemp'] > -2]

df['minTemp'].plot(color='r', label='MinTemp')
df['maxTemp'].plot(color='b', label='MaxTemp')
plt.ylabel('Temperature in Degree Celsius')
plt.title('MinTemp and MaxTemp vs Date')
plt.xlabel('Date')
plt.grid()
plt.legend()
plt.show()

# Just from looking at the plot, the data looks stationary.
# We will plot the rolling mean and variance plot to know more about it.
# And finally we will do the statistical testing to confirm our findings.

# Rolling mean and variance plots

check_stationarity(df['temp'], 'temperature')
# from the statistical tests, we confirm that our data is stationary.

y = df['temp']
lags = 50
ACF_PACF_Plot(y, lags)

lags = 500
ACF_PACF_Plot(y, lags)

# from the ACF and PACF plot we can see that
# ACF is tailing off and PACF is cutoff at order 1
# so it means that it is an AR process with order 1.


ry = cal_autocorr(df['temp'], 800, 'mean Temperature')
# ymin = cal_autocorr(df['minTemp'], 50, 'Min Temperature')
# ymax = cal_autocorr(df['maxTemp'], 50, 'Max Temperature')
# ACF plot came as expected. Generally the temperature will depend on the past values too much
# And ACF will decay as the number of lags are increased.

STL_analysis(y, periods=365)

# # as expected, seasonality is very high, and trend is very low when period is set to 365

# splitting data into test and train
X = df.drop(['temp', 'minTemp', 'maxTemp'], axis=1)
y = df['temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print('Length of X_train and y_train:', len(X_train))
print('Length of X_test and y_test:', len(X_test))

# Holt Winter Method
# this model is just considering 'temp'
# we will consider 'minTemp', 'maxTemp', and 'humidity' later
modelES = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=365).fit()

forecastES = modelES.forecast(steps=len(y_test))

title = 'Winter-Holt forecasting'
plot_forecasting_models(y_train, y_test, forecastES, title)

# Calculating MSE for Winter-Holt method
_, _, mse = cal_error_MSE(y_test, forecastES)

rmse_winter = np.sqrt(mse)
mae_winter = mean_absolute_error(y_test, forecastES)
print('MSE for Winter-Holt method:', np.round(mse, 2))
print('RMSE for Winter-Holt method:', np.round(np.sqrt(mse), 2))
print("MAE for Holt-Winter method:", np.round(mae_winter, 2))


# Now will forecast using base models and plot them in a subplot

# Average forecasting
_, forecast_average = average_forecasting(y_train, y_test)
_, _, mse_average = cal_error_MSE(y_test, forecast_average)
rmse_average = np.sqrt(mse_average)
mae_average = mean_absolute_error(y_test, forecast_average)
print('MSE for Average forecasting:', np.round(mse_average, 2))
print('RMSE for Average forecasting:', np.round(rmse_average, 2))
print("MAE for Average forecasting:", np.round(mae_average, 2))


# Naive forecasting
_, forecast_Naive = Naive_forecasting(y_train, y_test)
_, _, mse_Naive = cal_error_MSE(y_test, forecast_Naive)
rmse_Naive = np.sqrt(mse_Naive)
mae_Naive = mean_absolute_error(y_test, forecast_Naive)
print('MSE for Naive forecasting:', np.round(mse_Naive, 2))
print('RMSE for Naive forecasting:', np.round(rmse_Naive, 2))
print("MAE for Naive forecasting:", np.round(mae_Naive, 2))

# Drift forecasting
_, forecast_Drift = drift_forecasting(y_train, y_test)
_, _, mse_Drift = cal_error_MSE(y_test, forecast_Drift)
rmse_Drift = np.sqrt(mse_Drift)
mae_Drift = mean_absolute_error(y_test, forecast_Drift)
print('MSE for Drift forecasting:', np.round(mse_Drift, 2))
print('RMSE for Drift forecasting:', np.round(rmse_Drift, 2))
print("MAE for Drift forecasting:", np.round(mae_Drift, 2))

# Simple Exponential Smoothing
L0 = y_train[0]
_, forecast_SES = ses(y_train, y_test, L0, alpha=0.9)
_, _, mse_SES = cal_error_MSE(y_test, forecast_SES)
rmse_SES = np.sqrt(mse_SES)
mae_SES = mean_absolute_error(y_test, forecast_SES)
print('MSE for SES forecasting:', np.round(mse_SES, 2))
print('RMSE for SES forecasting:', np.round(rmse_SES, 2))
print("MAE for SES forecasting:", np.round(mae_SES, 2))

# subplots of the base models
fig = plt.figure(figsize=(16, 8))
axs = fig.subplots(2, 2)
plot_forecasting_models(y_train, y_test, forecast_average, 'Average Forecasting', axs=axs[0][0])
plot_forecasting_models(y_train, y_test, forecast_Naive, 'Naive Forecasting', axs=axs[0][1])
plot_forecasting_models(y_train, y_test, forecast_Drift, 'Drift Forecasting', axs=axs[1][0])
plot_forecasting_models(y_train, y_test, forecast_SES, 'SES Forecasting', axs=axs[1][1])
plt.tight_layout()
plt.show()
# # MSE for Winter-Holt method: 15.66
# # MSE for Average forecasting: 51.73
# # MSE for Naive forecasting: 63.07
# # MSE for Drift forecasting: 70.36
# # MSE for SES forecasting: 51.11

# Lowest MSE till now is of Winter-Holt seasonal method

# Scaling the data using standard Scaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled, X_test_scaled = standardize(X_train, X_test)

condition_number = LA.cond(X_train_scaled)
print('condition_number:', condition_number)
# From the condition number we can see that there is severe degree of collinearity in our dataset

X_svd = X_train.to_numpy()
H = np.matmul(X_svd.T, X_svd)

s, d, v = LA.svd(H)
print('SingularValues =', d)
# SingularValues = [6.04547993e+13 2.70315928e+08 6.72024236e+06 3.02061569e+05
#  1.55522232e+05 2.52494063e+04 1.05676115e+02 7.94014223e+01
#  1.54329575e+01 1.27133958e-01 1.81119775e-02 4.14548803e-15]

# From the singular values, there are values which are very close to zero.
# So, it means that there is some collinearity in our dataset.

# so, it 100 percent sure that collinearity exists in our dataset.
# We need to fix it.

# adding bias
X_train_scaled = sm.add_constant(X_train_scaled)

cols = X_train.columns
cols = np.insert(cols, 0, 'constant')
# Going forward with Backward Stepwise Regression to reduce features
X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
X_test_scaled = pd.concat([pd.Series(1, index=X_test.index, name='constant'), X_test_scaled], axis=1)
result = sm.OLS(y_train, X_train_scaled).fit()
print(result.summary())

# p-value of tornado is very high which means it is insignificant
# removing that

# X_train_scaled.drop(['tornado'], axis=1, inplace=True)
# result = sm.OLS(y_train, X_train_scaled).fit()
# print(result.summary())
#
# X_train_scaled.drop(['tornado', 'hail'], axis=1, inplace=True)
# result = sm.OLS(y_train, X_train_scaled).fit()
# print(result.summary())
#
# # removing pressure
# X_train_scaled.drop(['tornado', 'hail', 'pressure'], axis=1, inplace=True)
# result = sm.OLS(y_train, X_train_scaled).fit()
# print(result.summary())
#
# # removing visibility
# X_train_scaled.drop(['tornado', 'hail', 'pressure', 'visibility'], axis=1, inplace=True)
# result = sm.OLS(y_train, X_train_scaled).fit()
# print(result.summary())
#
# # removing snow
# X_train_scaled.drop(['tornado', 'hail', 'pressure', 'visibility', 'snow'], axis=1, inplace=True)
# result = sm.OLS(y_train, X_train_scaled).fit()
# print(result.summary())
#
# # removing thunder
# X_train_scaled.drop(['tornado', 'hail', 'pressure', 'visibility', 'snow', 'thunder'], axis=1, inplace=True)
# result = sm.OLS(y_train, X_train_scaled).fit()
# print(result.summary())
#
# # # removing rain
# X_train_scaled.drop(['tornado', 'hail', 'pressure', 'visibility', 'snow', 'thunder', 'rain'], axis=1, inplace=True)
# result = sm.OLS(y_train, X_train_scaled).fit()
# print(result.summary())

# removing fog
X_train_scaled.drop(['tornado', 'hail', 'pressure', 'visibility', 'snow', 'thunder', 'rain', 'fog'], axis=1, inplace=True)
X_test_scaled.drop(['tornado', 'hail', 'pressure', 'visibility', 'snow', 'thunder', 'rain', 'fog'], axis=1, inplace=True)
result = sm.OLS(y_train, X_train_scaled).fit()
print(result.summary())

pred_ols = result.predict(X_test_scaled)
plot_forecasting_models(y_train, y_test, pred_ols, 'OLS Model')
mse_ols = mean_squared_error(y_test, pred_ols)
rmse_ols = np.sqrt(mse_ols)
mae_ols = mean_absolute_error(y_test, pred_ols)
print('MSE for OLS model:', np.round(mse_ols, 2))
print('RMSE for OLS model:', np.round(rmse_ols, 2))
print("MAE for OLS model:", np.round(mae_ols, 2))
ols_residuals = result.resid
q_ols = cal_Q_value(ols_residuals, 'OLS Residuals', 50)
print('Q Value of OLS residuals:', np.round(q_ols, 2))
# 42426.55565814915
print('Mean of residuals for OLS:', np.mean(ols_residuals))
print('Variance of residuals for OLS:', np.var(ols_residuals))

# Finding Ry values and calling GPAC


y_diff = differencing(y_train, 1, 365)  # s = 365 for seasonal differencing
y_diff = np.array(removeNA(y_diff))
y_diff = y_diff.astype(float)

# y = y_diff
ry = cal_autocorr(y_diff, 100, 'temp')

cal_gpac(ry, 10, 10)
# from the GPAC, the most probable order is ARMA(1, 0) and (1, 4)
na = 7
nb = 0

theta_hat, variance_hat, covariance_hat, sse_array = LM(y_diff, na, nb)
theta_hat = theta_hat.ravel()

print('Estimated parameters')
print(theta_hat)
print()
print('Estimated parameters with 95% confidence interval')
for i in range(na):
    print(f'{theta_hat[i]-(2*np.sqrt(covariance_hat[i][i]))} < a{i+1} < {theta_hat[i]+(2*np.sqrt(covariance_hat[i][i]))}')
print()
j = 0
for i in range(na, na+nb):
    print(f'{theta_hat[i] - (2 * np.sqrt(covariance_hat[i][i]))} < b{j + 1} < {theta_hat[i] + (2 * np.sqrt(covariance_hat[i][i]))}')
    j += 1

print('\nEstimated covariance matrix')
print(covariance_hat)
print()

print('Estimated variance')
print(variance_hat)
print()

print('Estimated standard deviation')
print(np.sqrt(variance_hat))
print()

sse_array = np.array(sse_array)
den, num = num_den(theta_hat, na, nb)
if num[1] != 0:
    print('Roots of the numerator are:', np.roots(num).real)

if den[1] != 0:
    print('Roots of the denominator are:', np.roots(den).real)

# Since, there is only one root of the process, there is no
# zero pole cancellation

x = np.arange(len(sse_array))
plt.plot(x, sse_array)
plt.xticks(np.arange(x[0], x[-1]+1, 1))
plt.xlabel('No. of iterations')
plt.ylabel('SSE')
plt.grid()
plt.title('SSE vs no. of iterations')
plt.show()

# y_diff = differencing(y, 1, s=365)
model_ARIMA, model_hat = prediction(y_diff, na, nb)
print(model_ARIMA.summary())

prediction1 = model_ARIMA.forecast(len(y_test))
prediction1 = pd.Series(prediction1, index=y_test.index)
y_arima_package_hat = reverse_transform_and_plot(prediction1, y_train, y_test, 'ARIMA package')
mse_arima_package = mean_squared_error(y_test, y_arima_package_hat)
print('MSE of ARIMA using package with order (7, 0):', mse_arima_package)
rmse_arima_package = np.sqrt(mse_arima_package)
mae_arima_package = mean_absolute_error(y_test, y_arima_package_hat)
print('MSE for ARIMA:', np.round(mse_arima_package, 2))
print('RMSE for ARIMA:', np.round(rmse_arima_package, 2))
print("MAE for ARIMA:", np.round(mae_arima_package, 2))


y_hat_test = forecast(y_diff, len(y_diff), len(y_test))
y_hat_test = pd.Series(y_hat_test)
y_hat_test.index = y_test.index

y_arima_custom_hat = reverse_transform_and_plot(y_hat_test, y_train, y_test, 'Custom Forecast function')
mse_custom = mean_squared_error(y_test, y_arima_custom_hat)
print('Variance of residual error:', np.round(variance_hat[0][0], 2))
print('Variance of forecast error:', np.round(np.var(y_test - y_arima_custom_hat), 2))

print('MSE of ARIMA using custom function with order (7, 0):', mse_custom)

# till now the best deal is for the one seasonal differencing of s=365 and na=2, nb=0
# this model is working best for me till now.
# I will just proceed with the deep learning models now

# ry = cal_autocorr(y_diff, 100, 'diff')
# cal_gpac(ry, 10, 10)
# model_hat = prediction(y_diff, 1, 4)
# ACF_PACF_Plot(y_diff, 10)
# model_hat = prediction(y_diff, 1, 4)
# e = y_diff - model_hat
# rye = cal_autocorr(e, 100, 'error')
# cal_gpac(rye, 8, 8)
# ACF_PACF_Plot(y_diff, 25)



# Trying for whole dataset
df2 = df.copy()
df2 = df2.drop(['minTemp', 'maxTemp'], axis=1)
col_to_move = 'temp'
last_col = df2.pop(col_to_move)
df2.insert(len(df2.columns), col_to_move, last_col)

scaler = StandardScaler()
scaler = scaler.fit(df2)
scaled_data = scaler.transform(df2)
train_size = int(len(df2) * 0.8)
train_data = scaled_data[:train_size, :]
time_step = 365
train_X, train_y = [], []
n_past = len(y) - train_size
for i in range(n_past, len(train_data)):
    train_X.append(train_data[i - time_step:i, 0:train_data.shape[1]-1])
    train_y.append(train_data[i, train_data.shape[1]-1])

train_X = np.array(train_X)
train_y = np.array(train_y)

model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(train_X, train_y,  batch_size=16, validation_split=.1, epochs=10, verbose=1)
plt.figure()
plt.plot(history.history['loss'], 'r', label='Training loss')
plt.plot(history.history['val_loss'], 'b', label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.title('Loss VS Iterations')
plt.legend()
plt.show()
dataset = df2.values
test_data = scaled_data[train_size-365:, :]
x_test = []
y_test = dataset[train_size:, -1]

for i in range(365, len(test_data)):
    x_test.append(test_data[i-365:i, 0:train_data.shape[1]-1])

joblib.dump(model, 'LSTM_model.pkl')

model = joblib.load('LSTM_model.pkl')
x_test = np.array(x_test)
predictions = model.predict(x_test)
forecast_copies = np.repeat(predictions, train_X.shape[2]+1, axis=-1)
predictions = scaler.inverse_transform(forecast_copies)[:, -1]

train = df2.iloc[:train_size]
valid = df2.iloc[train_size:]
valid['Predictions'] = predictions

fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
ax.set_title('Temperature prediction in Delhi using LSTM')
ax.set_xlabel("Date", fontsize=18)
ax.set_ylabel('Temperature')
ax.plot(train['temp'], 'blue')
ax.plot(valid['temp'],  'red')
ax.plot(valid['Predictions'], 'black')
ax.legend(["Train", "Val", "Predictions"], loc="lower right", fontsize=18)
ax.grid()
plt.show()
mse_lstm = mean_squared_error(valid['temp'], valid['Predictions'])
rmse_lstm = np.sqrt(mse_lstm)
mae_lstm = mean_absolute_error(valid['temp'], valid['Predictions'])
print("MSE for LSTM model:", np.round(mse_lstm, 2))
print("RMSE for LSTM model:", np.round(rmse_lstm, 2))
print("MAE for LSTM model:", np.round(mae_lstm, 2))


X = df.drop(['temp', 'minTemp', 'maxTemp'], axis=1)
y = df['temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

fig, axs = plt.subplots(2, 2, figsize=(16, 10))
axs[0, 0].plot(train['temp'], 'blue')
axs[0, 0].plot(valid['temp'],  'red')
axs[0, 0].plot(valid['Predictions'], 'black')
axs[0, 0].set_title('LSTM')
axs[0, 0].set_xlabel('Time')
axs[0, 0].set_ylabel('Temperature')

axs[0, 1].plot(y_train.index, y_train.values, label='Train')
axs[0, 1].plot(y_arima_custom_hat.index, y_arima_custom_hat.values, label='Forecast')
axs[0, 1].plot(y_test.index, y_test.values, label='Actual Test Data')
axs[0, 1].set_title('ARIMA')
axs[0, 1].set_xlabel('Time')
axs[0, 1].set_ylabel('Temperature')

axs[1, 0].plot(y_train.index, y_train, color='r', label='train')
axs[1, 0].plot(y_test.index, y_test, color='g', label='test')
axs[1, 0].plot(y_test.index, forecastES, color='b', label='h step')
axs[1, 0].set_title('Winter Holt')
axs[1, 0].set_xlabel('Time')
axs[1, 0].set_ylabel('Temperature')

axs[1, 1].plot(y_train.index, y_train, color='r', label='train')
axs[1, 1].plot(y_test.index, y_test, color='g', label='test')
axs[1, 1].plot(y_test.index, pred_ols, color='b', label='h step')
axs[1, 1].set_title('OLS')
axs[1, 1].set_xlabel('Time')
axs[1, 1].set_ylabel('Temperature')
fig.suptitle('All the models performance')

plt.tight_layout()
plt.show()



# y_test, y_arima_custom_hat

plt.plot(y_test, label='Test')
plt.plot(y_arima_custom_hat, label='Forecast using ARIMA')
plt.title('h-step Predictions on Test Set')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.tight_layout()
plt.show()
