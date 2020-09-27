# Example 15.3 ARCH Analysis of U.S. Inflation

import numpy as np
import pandas as pd
import statsmodels.api as sm

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

from statsmodels.tsa.arima_model import ARMA
ar3 = ARMA(dp,order=(3,0)).fit(method='mle')
print(ar3.summary())
# check for serial correlation in residuals
# make sure Box_test and ACF_test is defined (see example15_1)
# make sure tsplot is defined (see example15_2)
print(Box_test(ar3.resid,12))
print(ACF_test(ar3.resid,12))
tsplot(ar3.resid,lags=12)
# time series analysis: diagnostic checking for volatility
tsplot(ar3.resid**2,lags=12)

# GARCH(1,1), a common model
from arch import arch_model
arch1 = arch_model(dp,mean='AR',lags=3,vol='Garch',p=1,q=1).fit()
print(arch1.summary())
arch1.plot()
arch1.arch_lm_test(lags=10,standardized=True)

# model building blocks: mean, variance, distribution
from arch.univariate import ARX,GARCH,Normal

# GARCH(1,1)
model1 = ARX(dp,lags=3)
model1.volatility = GARCH(p=1,q=1)
model1.distribution = Normal()
# note: p=#ARCH (lags of squared errors), q=#GARCH (lags of variances)
garch1 = model1.fit()
print(garch1.summary())
# garch1.resid and garch1.conditional_volatility contain NaN
# garch1.arch_lm_test(lags=10,standardized=True) can not be used
v = garch1.conditional_volatility.dropna()
e = garch1.resid.dropna()/np.sqrt(v)
tsplot(e,lags=12)
