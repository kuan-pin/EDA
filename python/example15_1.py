# Example 15.1 ARMA Analysis of Bond Yields

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import matplotlib.pyplot as plt
%matplotlib inline

def Box_test(y,lags=10):
    lb, lbpv, bp, bppv = sm.stats.diagnostic.acorr_ljungbox(y,lags,True)
    boxtest = pd.DataFrame({'Ljung-Box': lb, 'LB-P-val': lbpv,
                            'Box-Pierce': bp, 'BP-P-val': bppv},
                            index=range(1,lags+1))
    return (boxtest)

def ACF_test(y,lags=10):    
    acf, acf_confint, lbq, lbqpv = tsa.acf(y, nlags=lags, alpha=0.05, qstat=True, fft=False)
    pacf, pacf_confint = tsa.pacf(y, nlags=lags, alpha=0.05)
    acf_se = (acf_confint[:,1]-acf)/1.96
    pacf_se = (pacf_confint[:,1]-pacf)/1.96
    acftest = pd.DataFrame({'AR': acf[1:], 'AR_se': acf_se[1:], 
                            'PAR': pacf[1:], 'PAR_se': pacf_se[1:]},
                            index=range(1,lags+1))
    return(acftest)

def ACF_plot(y,lags=10):
    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(2,1,1)
    fig = sm.graphics.tsa.plot_acf(y, ax=ax1, lags=lags, zero=False)
    ax2 = fig.add_subplot(2,1,2)
    fig = sm.graphics.tsa.plot_pacf(y, ax=ax2, lags=lags, zero=False)
    plt.show()
    return
    
data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/bonds.txt",sep='\s+',nrows=60)
# setup time series index
Year = (100*data.date)//100
Month = (100*data.date)%100
data.index = pd.PeriodIndex(year=Year,month=Month,freq='M')

ols1 = sm.OLS.from_formula('Y~Y.shift(1)+Y.shift(2)',data=data,missing='drop').fit()
print(ols1.summary())
# check for residual autocorrelation
print(Box_test(ols1.resid,4))
print(ACF_test(ols1.resid,12))

from statsmodels.tsa.ar_model import AutoReg
ar2 = AutoReg(data.Y,lags=2).fit()
print(ar2.summary())
# check for residual autocorrelation
print(Box_test(ar2.resid,4))
print(ACF_test(ar2.resid,12))
ACF_plot(ar2.resid,20)

from statsmodels.tsa.arima_model import ARMA
arma20 = ARMA(data.Y,order=(2,0)).fit()
# arma1 = ARMA(data.Y,order=(2,0)).fit(method='css')
print(arma20.summary())
print(Box_test(arma20.resid,4))
print(ACF_test(arma20.resid,12))
ACF_plot(arma20.resid,20)

arma11 = ARMA(data.Y,order=(1,1)).fit()
print(arma11.summary())
print(Box_test(arma11.resid,4))
print(ACF_test(arma11.resid,12))
ACF_plot(arma11.resid,20)
