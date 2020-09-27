# Example 15.2 ARMA Analysis of U. S. Inflation

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import scipy.stats as stats
import matplotlib.pyplot as plt
%matplotlib inline

# the following code was adapted from the blog Seanabu.com
def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        tsa.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        tsa.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        stats.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 

data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/usinf.txt",sep='\s+',nrows=136)
# setup time series index
Year = (10*data.Qtr)//10
Quarter = (10*data.Qtr)%10
data.index = pd.PeriodIndex(year=Year,quarter=Quarter,freq='Q')
y = np.log(data.Y)
m = np.log(data.M1)
p = np.log(data.P)
# changes are scaled by multiplying 100
dy = 100*y.diff().dropna()
dm = 100*m.diff().dropna()
dp = 100*p.diff().dropna()
dmy = dm-dy

# time series analysis: identification
tsplot(dp,lags=12)
# time series analysis: estimation
from statsmodels.tsa.arima_model import ARMA
arma1 = ARMA(dp,order=(1,3),exog=dmy).fit()
print(arma1.summary())
# check for serial correlation in residuals
# make sure Box_test and ACF_test is defined (see example15_1)
print(Box_test(arma1.resid,12))
print(ACF_test(arma1.resid,12))
tsplot(arma1.resid,lags=12)

X = pd.concat([dmy.shift(1),dp.shift(1)],axis=1)
X.columns = ['L1.dmy','L1.dp']
arma2 = ARMA(dp,order=(0,3),exog=X,missing='drop').fit()
print(arma2.summary())
# check for serial correlation in residuals
print(Box_test(arma2.resid,12))
print(ACF_test(arma2.resid,12))
tsplot(arma2.resid,lags=12)

# alternatively, for model identification
# import statsmodels.graphics.tsaplots as tsaplt
# tsaplt.plot_acf(dp, lags=12)
# tsaplt.plot_pacf(dp, lags=12)
