# Example 17.1: Ex-Post Forecasts and Forecast Error Statistics
# In this example, we do not claim the best forecast based on the historical data. 
# We demonstrate the techniques of forecasting and evaluation.
# 
import numpy as np
import pandas as pd

# read data from 1959.1 to 2003.4 (181 obs.)
z = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/gdp96.txt',sep='\s+')
# setup time series index
z.index = pd.period_range('1959Q1', '2003Q4', freq='Q-DEC')

rgdp = 100*z['GDP']/z['PGDP2000']
growth = 100*(rgdp/rgdp.shift(4)-1.0)
xvar = pd.concat([z['LEADING96'].shift(1),z['LEADING96'].shift(5)],axis=1)

growth_train = growth['1961Q1':'2001Q4']
xvar_train = xvar['1961Q1':'2001Q4']
growth_test = growth['2002Q1':]
xvar_test = xvar['2002Q1':]

from statsmodels.tsa.arima_model import ARMA
arma1 = ARMA(growth_train,exog=xvar_train,order=(1,4)).fit()
print(arma1.summary())

import matplotlib.pyplot as plt 
%matplotlib inline
for1 = arma1.predict(start='2002Q1',end='2003Q4',exog=xvar_test)
for1.plot()
plt.plot(growth_test)
plt.show()
print(for1)

for2 = arma1.forecast(steps=8,exog=xvar_test,alpha=0.05)
for2[0]  # forecast value
for2[1]  # forecast se
for2[2]  # forecast interval
for2data ={'actural':growth_test, 
           'forecast':for2[0],
           's.e.':for2[1],
           'lower':for2[2][:,0],
           'upper':for2[2][:,1]}
for2index = pd.period_range(start='2002Q1',end='2003Q4',freq='Q-DEC')
# create forecast data frame
forecast = pd.DataFrame(data=for2data,index=for2index)
forecast.plot(y=['actural','forecast','lower','upper'])
print(forecast)

# forecast_error_statistics takes a forecast dataframe with
# actural and forecast data columns
def forecast_error_statistics(f):
    x = f.actural
    p = f.forecast
    e = x-p
    mx = x.mean()
    mp = p.mean()
    sx = x.std(ddof=0)
    sp = p.std(ddof=0)
    r = x.corr(p)
    r2 = r**2
    ME = e.mean()
    MAE = (e.abs()).mean()
    MAPE = (100*((e/x).abs())).mean()
    MSE = (e**2).mean()
    RMSE = np.sqrt(MSE)
    MSPE = 100*(((e/x)**2).mean())
    RMSPE = 100*np.sqrt(((e/x)**2).mean())
    # MSE Decomposition
    Um = ((mx-mp)**2)/MSE
    Us = ((sx-sp)**2)/MSE
    Uc = (2*(1-r)*sp*sx)/MSE
    Ur = ((sp-r*sx)**2)/MSE
    Ud = ((1-r**2)*sx**2)/MSE
    res1data = {'Statistics':[r2,ME,MAE,MSE,RMSE,MAPE,RMSPE]}
    res1index = ['R-Square Between Observed and Predicted',
             'Mean Error (ME)',
             'Mean Absolute Error (MAE)',
             'Mean Squared Error (MSE)',
             'Root Mean Squared Error (RMSE)',
             'Mean Absolute Percent Error (MAPE)',
             'Root Mean Squared Percent Error (RMSPE)']
    res1 = pd.DataFrame(data=res1data, index=res1index)
    res2data = {'Decomposition of MSE':[Um,Us,Uc,Ur,Ud]}
    res2index = ['Proportion Due to Bias',
             'Proportion Due to Variance',
             'Proportion Due to Covariance',
             'Proportion Due to Regression',
             'Proportion Due to Disturbance']
    res2 = pd.DataFrame(data=res2data, index=res2index)
    return(res1,res2)

stat1, stat2 = forecast_error_statistics(forecast)
print(stat1)
print(stat2)
