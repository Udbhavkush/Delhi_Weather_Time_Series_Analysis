import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import statsmodels.api as sm
import seaborn as sns
import numpy.linalg as LA
from scipy import signal
from scipy.stats import chi2
from statsmodels.tsa.seasonal import STL


def plot_data(df, title='', xlab='Time', ylab=''):
    df.plot()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    # plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def import_data(url, title='', xlab='Time', ylab=''):
    df = pd.read_csv(url, header=0, parse_dates=[0], index_col=0)
    plot_data(df, title, xlab, ylab)
    return df

# def cal_rolling_mean_var(df, col):
#     roll_mean = df.mean()
#     roll_variance = df.var()
#     if pd.isna(roll_mean):
#         roll_mean = 0
#     if pd.isna(roll_variance):
#         roll_variance = 0
#     return [roll_mean, roll_variance]


def cal_rolling_mean_var2(y):
    roll_mean = y.mean()
    roll_variance = y.var()
    if pd.isna(roll_mean):
        roll_mean = 0
    if pd.isna(roll_variance):
        roll_variance = 0
    return [roll_mean, roll_variance]


def plot_rolling_mean_var(y, col):
    rolling_mean = []
    rolling_variance = []
    for i in range(1, len(y) + 1):
        rolling_values = cal_rolling_mean_var2(y.head(i))
        rolling_mean.append(rolling_values[0])
        rolling_variance.append(rolling_values[1])

    fig, axs = plt.subplots(2)
    axs[0].plot(rolling_mean)
    axs[0].set(ylabel='Magnitude', xlabel='samples', title='Rolling mean-' + col)
    axs[1].plot(rolling_variance)
    axs[1].set(ylabel='Magnitude', xlabel='samples', title='Rolling variance-' + col)
    axs[0].legend(["Varying Mean"], loc='lower right')
    axs[1].legend(["Varying Variance"], loc='lower right')
    plt.tight_layout()
    plt.show()


def rolling_mean_var(y, title):
    n = len(y)
    rolling_mean = np.zeros(len(y))
    rolling_var = np.zeros(len(y))
    for i in range(1, n+1):
        tempy = np.zeros(i)
        for j in range(i):
            tempy[j] = y[j]
        rolling_mean[i-1], rolling_var[i-1] = cal_rolling_mean_var(tempy, title)

    fig, axs = plt.subplots(2)
    axs[0].plot(rolling_mean)
    axs[0].set(ylabel='Magnitude', xlabel='samples', title='Rolling mean-' + title)
    axs[1].plot(rolling_var)
    axs[1].set(ylabel='Magnitude', xlabel='samples', title='Rolling variance-' + title)
    axs[0].legend(["Varying Mean"], loc='lower right')
    axs[1].legend(["Varying Variance"], loc='lower right')
    plt.tight_layout()
    plt.show()


def ADF_Cal(x):
    x = x.fillna(0)
    print('\nADF TEST')
    print('NULL Hypothesis: Unit root is present i.e., time series is not stationary.')
    print('ALTERNATE Hypothesis: unit root is not present i.e., time series is stationary.')
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] < 0.05:
        print('Rejecting the NULL hypothesis with more than 95% confidence interval')
        print('Time series is stationary')
    else:
        print('Cannot reject the NULL hypothesis with 95% confidence interval')
        print('Time series is non-stationary')

# For ADFuller test, NULL and ALTERNATE hypothesis are as follows:
# NULL Hypothesis: Unit root is present i.e., time series is not stationary.
# ALTERNATE Hypothesis: unit root is not present i.e., time series is stationary.


def kpss_test(timeseries):
    timeseries = timeseries.fillna(0)
    print('\nKPSS TEST')
    print('NULL Hypothesis: Time series is stationary.')
    print('ALTERNATE Hypothesis: Time series is not stationary.')
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
    if kpss_output[1] < 0.05:
        print('Rejecting the NULL hypothesis with more than 95% confidence interval')
        print('Time series is non-stationary')
    else:
        print('Cannot reject the NULL hypothesis with 95% confidence interval')
        print('Time series is stationary')


# For KPSS test, NULL and ALTERNATE hypothesis are as follows:
# NULL Hypothesis: Time series is stationary.
# ALTERNATE Hypothesis: Time series is not stationary.

#
# def differencing1(df, col, s=1):
#     diff = []
#     n = len(df)
#     for i in range(s):
#         diff.append(None)
#     for i in range(s, n):
#         diff.append(df[col][i] - df[col][i - s])
#
#     return diff


def differencing(y, order, s=1):
    diff = []
    n = len(y)
    for i in range(n):
        if i-s < 0 or y[i] is None or y[i-s] is None:
            diff.append(None)
        else:
            diff.append(y[i] - y[i - s])
    if order == 1:
        return np.array(diff)
    return differencing(pd.Series(diff), order-1, s)


def check_stationarity(y, title):
    print('STATIONARITY CHECK')
    y = pd.Series(y.ravel())
    plot_rolling_mean_var(y, title)
    ADF_Cal(y)
    kpss_test(y)


def logTransform(y):
    return np.log(y)


# autocorrelation
def cal_autocorr(Y, lags, title, axs=None):  # default value is set to None, i.e. the case when we don't need subplots
    flag = True
    if axs is None:
        axs = plt
        flag = False

    T = len(Y)
    ry = []
    den = 0
    ybar = np.mean(Y)
    for y in Y:  # since denominator is constant for every iteration, we calculate it only once and store it.
        den = den + (y - ybar) ** 2

    for tau in range(lags+1):
        num = 0
        for t in range(tau, T):
            num = num + (Y[t] - ybar) * (Y[t - tau] - ybar)
        ry.append(num / den)

    ryy = ry[::-1]
    Ry = ryy[:-1] + ry  # to make the plot on both sides, reversed the list and added to the original list

    x = np.linspace(-lags, lags, 2 * lags + 1)
    markers, _, _ = axs.stem(x, Ry)
    plt.setp(markers, color='red', marker='o')
    axs.axhspan(-(1.96 / (T ** 0.5)), (1.96 / (T ** 0.5)), alpha=0.2, color='blue')

    if not flag:  # in this case, axs = plt, hence different functions to set xlabel, ylabel, and title
        axs.xlabel('Lags')
        axs.ylabel('Magnitude')
        axs.title(f'Autocorrelation plot of {title}')
        plt.show()
    else:
        axs.set_xlabel('Lags')
        axs.set_ylabel('Magnitude')
        axs.set_title(f'Autocorrelation plot of {title}')
        # in case of axes given i.e. we need subplots, we don't use plt.show() inside this function
        # as it will plot every subplot separately. We need to use plt.show() outside the loop from where
        # the function is called when we need subplots.
    return ry

# to call it for different axs values, this is the reference code
# fig = plt.figure(figsize=(16, 8))
# axs = fig.subplots(3)
# cal_autocorr(y, lags, '1000 samples', axs[0])
#
# N = 10000
# y_10000, _ = process_MA(N)
# cal_autocorr(y_10000, lags, '10000 samples', axs[1])
#
# N = 100000
# y_100000, _ = process_MA(N)
# cal_autocorr(y_100000, lags, '100000 samples', axs[2])
#
# plt.tight_layout()
# plt.show()


def cal_error_MSE(y, yhat, skip=0):
    y = np.array(y)
    yhat = np.array(yhat)
    error = []
    error_square = []
    n = len(y)
    for i in range(n):
        if yhat[i] is None:
            error.append(None)
            error_square.append(None)
        else:
            error.append(y[i]-yhat[i])
            error_square.append((y[i]-yhat[i])**2)
    mse = 0
    for i in range(skip, n):
        mse = mse + error_square[i]

    mse = mse/(n-skip)

    return error, error_square, np.round(mse, 2)


def plot_forecasting_models(ytrain, ytest, yhatTest, title, axs=None):
    if axs is None:
        axs = plt
    x = np.arange(1, len(ytrain)+len(ytest)+1)
    x1 = x[:len(ytrain)]
    x2 = x[len(ytrain):]
    axs.plot(ytrain.index, ytrain, color='r', label='train')
    axs.plot(ytest.index, ytest, color='g', label='test')
    axs.plot(ytest.index, yhatTest, color='b', label='h step')
    # axs.plot(np.arange(len(ytrain)), ytrain, color='r', label='train')
    # axs.plot(np.arange(len(ytrain), len(ytrain)+len(ytest)), ytest, color='g', label='test')
    # axs.plot(np.arange(len(ytrain), len(ytrain)+len(yhatTest)), yhatTest, color='b', label='h step')
    if axs is plt:
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        axs.set_xlabel('Time')
        axs.set_ylabel('Values')
        axs.set_title(title)
        axs.grid()
        axs.legend()


# def average_forecasting(ytrain, ytest):
#     n = len(ytrain)
#     yhatTrain = []
#     yhatTest = []
#     yhatTrain.append(None)
#     for i in range(1, n):
#         mean = np.mean(ytrain[0:i])
#         yhatTrain.append(np.round(mean, 2))
#
#     mean = np.mean(ytrain)
#     n = len(ytest)
#     for i in range(n):
#         yhatTest.append(np.round(mean, 2))
#
#     return yhatTrain, yhatTest

def average_forecasting(ytrain, ytest):
    n = len(ytrain)
    yhatTrain = ytrain.copy()
    yhatTest = ytest.copy()
    yhatTrain[0] = None
    for i in range(1, n):
        mean = np.mean(ytrain[0:i])
        yhatTrain[i] = np.round(mean, 2)
        # yhatTrain.append(np.round(mean, 2))

    mean = np.mean(ytrain)
    n = len(ytest)
    for i in range(n):
        yhatTest[i] = np.round(mean, 2)
        # yhatTest.append(np.round(mean, 2))

    return yhatTrain, yhatTest


# def Naive_forecasting(xtrain, xtest):
#     n = len(xtrain)
#     yhatTrain = []
#     yhatTest = []
#     yhatTrain.append(None)
#     for i in range(1, n):
#         yhatTrain.append(xtrain[i-1])
#
#     yT = xtrain[n-1]
#     n = len(xtest)
#     for i in range(n):
#         yhatTest.append(yT)
#
#     return yhatTrain, yhatTest

def Naive_forecasting(xtrain, xtest):
    n = len(xtrain)
    yhatTrain = xtrain.copy()
    yhatTest = xtest.copy()
    yhatTrain[0] = None
    for i in range(1, n):
        yhatTrain[i] = xtrain[i-1]
        # yhatTrain.append(xtrain[i-1])

    yT = xtrain[n-1]
    n = len(xtest)
    for i in range(n):
        yhatTest[i] = yT

    return yhatTrain, yhatTest


# def drift_forecasting(xtrain, xtest):
#     n = len(xtrain)
#     yhatTrain = []
#     yhatTest = []
#     yhatTrain.append(None)
#     yhatTrain.append(None)
#     for i in range(2, n):
#         y = xtrain[i-1] + ((xtrain[i-1] - xtrain[0])/(i-1))
#         yhatTrain.append(y)
#
#     slope = (xtrain[n-1] - xtrain[0])/(n-1)
#     y = xtrain[n-1]
#     n = len(xtest)
#     for i in range(1, n+1):
#         yhat = y + i * slope
#         yhatTest.append(yhat)
#
#     return yhatTrain, yhatTest

def drift_forecasting(xtrain, xtest):
    n = len(xtrain)
    yhatTrain = xtrain.copy()
    yhatTest = xtest.copy()
    yhatTrain[0] = None
    yhatTrain[1] = None
    for i in range(2, n):
        y = xtrain[i-1] + ((xtrain[i-1] - xtrain[0])/(i-1))
        yhatTrain[i] = y

    slope = (xtrain[n-1] - xtrain[0])/(n-1)
    y = xtrain[n-1]
    n = len(xtest)
    for i in range(1, n+1):
        yhat = y + i * slope
        yhatTest[i-1] = yhat

    return yhatTrain, yhatTest


# def ses(ytrain, ytest, L0, alpha=0.5):
#     n = len(ytrain)
#     yhatTrain = []
#     yhatTrain.append(L0)
#     for i in range(1, n):
#         yhat = alpha * ytrain[i-1] + (1-alpha) * yhatTrain[i-1]
#         yhatTrain.append(yhat)
#
#     yhatTest = []
#     l0 = alpha * ytrain[n-1] + (1-alpha) * yhatTrain[n-1]
#
#     n = len(ytest)
#     for i in range(n):
#         yhatTest.append(l0)
#
#     return yhatTrain, yhatTest

def ses(ytrain, ytest, L0, alpha=0.5):
    n = len(ytrain)
    yhatTrain = ytrain.copy()
    yhatTrain[0] = L0
    for i in range(1, n):
        yhat = alpha * ytrain[i-1] + (1-alpha) * yhatTrain[i-1]
        yhatTrain[i] = yhat

    yhatTest = ytest.copy()
    l0 = alpha * ytrain[n-1] + (1-alpha) * yhatTrain[n-1]

    n = len(ytest)
    for i in range(n):
        yhatTest[i] = l0

    return yhatTrain, yhatTest


def cal_Q_value(y, title, lags=5):
    # title = 'Average forecasting train data'
    acf = cal_autocorr(y, lags, title)
    sum_rk = 0
    T = len(y)
    for i in range(1, lags+1):
        sum_rk += acf[i]**2
    Q = T * sum_rk
    # if Q < Q* then white residual
    return Q


def standardize(train, test):
    columns = train.columns
    X_train = train.copy()
    X_test = test.copy()
    for col in columns:
        xbar = np.mean(X_train[col])
        std = np.std(X_train[col])
        X_train[col] = (X_train[col] - xbar) / std
        X_test[col] = (X_test[col] - xbar) / std

    return X_train, X_test


# 𝛽̂= (𝑋𝑇𝑋)−1𝑋𝑇𝑌
def normal_equation_LSE(X, Y):
    # X = x.to_numpy()
    # Y = y.to_numpy()
    normal_eqn = ((np.linalg.inv(X.T@X))@X.T)@Y
    return normal_eqn


def moving_average_decomposition(arr, order):
    m = order
    k = (m - 1) // 2
    res = []
    len_data = len(arr)

    if m == 2:
        res.append(None)
    else:
        for i in range(k):
            res.append(None)

    for i in range(len_data - m + 1):
        s = 0
        flag = True
        for j in range(i, i+m):
            if arr[j] is None:
                flag = False
                break
            s += arr[j]
        if flag is False:
            res.append(None)
        else:
            res.append(s/m)
    if m % 2 == 0 and m != 2:
        for i in range(k+1):
            res.append(None)
    elif m != 2:
        for i in range(k):
            res.append(None)

    return res


def create_process_general_AR(order, N, a):
    na = order
    np.random.seed(6313)
    mean = 0
    std = 1
    e = np.random.normal(mean, std, N)
    y = np.zeros(len(e))
    coef = np.zeros(na)
    for t in range(len(e)):
        sum_coef = 0
        y[t] = e[t]
        for i in range(1, na+1):
            if t-i < 0:
                break
            else:
                sum_coef += coef[i-1]*y[t-i]
        if t < na:
            coef[t] = a[t]

        y[t] -= sum_coef

    return y


def whitenoise(mean, std, samples, seed=0):
    np.random.seed(seed)
    return np.random.normal(mean, std, samples)


def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


def calc_val(Ry, J, K):

    den = np.zeros((K, K))

    for k in range(K):
        row = np.zeros(K)
        for i in range(K):
            row[i] = Ry[np.abs(J + k - i)]
        den[k] = row
    # num = den.copy()
    col = np.zeros(K)
    for i in range(K):
        col[i] = Ry[J+i+1]

    num = np.concatenate((den[:, :-1], col.reshape(-1, 1)), axis=1)
    num = np.array(num)
    den = np.array(den)

    if np.linalg.det(den) == 0:
        return np.inf
    if np.abs(np.linalg.det(num)/np.linalg.det(den)) < 0.00001:
        return 0
    return np.linalg.det(num)/np.linalg.det(den)


def cal_gpac(Ry, J=7, K=7):
    gpac_arr = np.zeros((J, K))
    gpac_arr.fill(None)
    for k in range(1, K):
        for j in range(J):
            gpac_arr[j][k] = calc_val(Ry, j, k)
    gpac_arr = np.delete(gpac_arr, 0, axis=1)
    # creating dataframe
    cols = []
    for k in range(1, K):
        cols.append(k)
    ind = []
    for j in range(J):
        ind.append(j)
    df = pd.DataFrame(gpac_arr, columns=cols, index=ind)

    fig = plt.figure()
    ax = sns.heatmap(df, annot=True, fmt='0.3f')  # cmap='Pastel2'
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.tight_layout()
    plt.show()
    print(df)


def check_AIC_BIC_adjR2(x, y):
    columns = x.columns
    res_df = pd.DataFrame(columns=['Removing column', 'AIC', 'BIC', 'AdjR2'])
    for col in columns:
        temp_df = x.copy()
        temp_df = temp_df.drop([col], axis=1)
        res = sm.OLS(y, temp_df).fit()
        res_df.loc[len(res_df.index)] = [col, res.aic, res.bic, res.rsquared_adj]

    res_df = res_df.sort_values(by=['AIC'], ascending=False)
    return res_df


# LM algo and supporting functions

def cal_e(num, den, y):
    system = (num, den, 1)
    _, e = signal.dlsim(system, y)
    return e


def num_den(theta, na, nb):
    theta = theta.ravel()
    num = np.concatenate(([1], theta[:na]))
    den = np.concatenate(([1], theta[na:]))
    max_len = max(len(num), len(den))
    num = np.pad(num, (0, max_len - len(num)), 'constant')
    den = np.pad(den, (0, max_len - len(den)), 'constant')
    return num, den


def cal_gradient_hessian(y, e, theta, na, nb):
    delta = 0.000001
    X = np.empty((len(e), 0))
    for i in range(len(theta)):
        temp_theta = theta.copy()
        temp_theta[i] = temp_theta[i] + delta
        num, den = num_den(temp_theta, na, nb)
        e_new = cal_e(num, den, y)
        x_temp = (e - e_new)/delta
        X = np.hstack((X, x_temp))

    # A = X.T @ X
    # g = X.T @ e
    A = np.dot(X.T, X)
    g = np.dot(X.T, e)
    return A, g


def SSE(theta, y, na, nb):
    num, den = num_den(theta, na, nb)
    e = cal_e(num, den, y)
    return np.dot(e.T, e)


def LM(y, na, nb):
    epoch = 0
    epochs = 50
    theta = np.zeros(na + nb)
    mu = 0.01
    n = len(theta)
    N = len(y)
    mu_max = 1e+20
    sse_array = []
    while epoch < epochs:
        sse_array.append(SSE(theta, y, na, nb).ravel())
        num, den = num_den(theta, na, nb)
        e = cal_e(num, den, y)
        A, g = cal_gradient_hessian(y, e, theta, na, nb)
        del_theta = LA.inv(A + mu*np.identity(A.shape[0])) @ g
        theta_new = theta.reshape(-1, 1) + del_theta
        sse_new = SSE(theta_new.ravel(), y, na, nb)
        sse_old = SSE(theta.ravel(), y, na, nb)
        if sse_new[0][0] < sse_old[0][0]:
            if LA.norm(del_theta) < 1e-3:
                theta_hat = theta_new.copy()
                sse_array.append(SSE(theta_new, y, na, nb).ravel())
                variance_hat = SSE(theta_new.ravel(), y, na, nb)/(N-n)
                covariance_hat = variance_hat * LA.inv(A)
                return theta_hat, variance_hat, covariance_hat, sse_array
            else:
                mu = mu/10
        while SSE(theta_new.ravel(), y, na, nb) >= SSE(theta.ravel(), y, na, nb):
            mu = mu*10
            # theta = theta_new.copy()
            if mu > mu_max:
                print('Error')
                break
            del_theta = LA.inv(A + mu * np.identity(A.shape[0])) @ g
            theta_new = theta.reshape(-1, 1) + del_theta
        epoch += 1
        theta = theta_new.copy()
    return


def removeNA(y):
    y = pd.Series(y)
    return y.dropna()


def STL_analysis(y, periods):
    y = pd.Series(y)
    stl = STL(y, period=periods)
    res = stl.fit()
    fig = res.plot()
    plt.suptitle('Trend, seasonality, and remainder plot')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

    T = res.trend
    S = res.seasonal
    R = res.resid

    plt.figure(figsize=[16, 8])
    plt.plot(y, label='Original')
    plt.plot(y - S, label='Seasonally adjusted')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Seasonally adjusted vs original curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=[16, 8])
    plt.plot(y, label='Original')
    plt.plot(y - T, label='Detrended')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Detrended vs original curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    Ft = max(0, 1 - np.var(R) / (np.var(T + R)))
    print("Strength of Trend for this dataset is ", Ft)

    seas = 1 - np.var(R) / (np.var(S + R))
    Fs = max(0, 1 - np.var(R) / (np.var(S + R)))
    print("Strength of seasonality for this dataset is ", Fs)


def prediction(y, na, nb):
    np.random.seed(6313)
    lags = 25
    N = len(y)
    y_var = np.var(y)
    # model = sm.tsa.statespace.SARIMAX(y, order=(0, 0, 0), seasonal_order=(2, 1, 0, 365)).fit()
    model = sm.tsa.arima.ARIMA(y, order=(na, 0, nb), trend=None).fit()
    model_hat = model.predict(start=0, end=N - 1)
    e = y - model_hat
    print('Lags', lags)
    re = cal_autocorr(np.array(e), lags, 'ACF of residuals')
    Q = len(y) * np.sum(np.square(re[1:]))
    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)
    print('Chi critical:', chi_critical)
    print('Q Value:', Q)
    print('Alfa value for 99% accuracy:', alfa)
    if Q < chi_critical:
        print("The residual is white ")
    else:
        print("The residual is NOT white ")
    plt.figure()
    plt.plot(y, 'r', label="True data")
    plt.plot(model_hat, 'b', label="Fitted data")
    plt.xlabel("Samples")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title(" Train versus One Step Prediction")
    plt.tight_layout()
    plt.show()

    return model, model_hat


def reverse_transform_and_plot(prediction, y_train, y_test, title):
    forecast = []
    s = 365
    for i in range(len(y_test)):
        if i < s:
            forecast.append(prediction[i] + y_train[- s + i])
        else:
            temp = i - s
            forecast.append(prediction[i] + forecast[temp])
    forecast = pd.Series(forecast)
    forecast.index = prediction.index
    plt.plot(y_train.index, y_train.values, label='Train')
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.plot(y_test.index, y_test.values, label='Actual Test Data')
    str = f'Predictions using {title}'
    plt.title(str)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return forecast