# Example 17.2: Ex-Ante Forecasts

import numpy as np
import pandas as pd

# read data from 1959.1 to 2003.4 (181 obs.)
z = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/gdp96.txt',sep='\s+')
# setup time series index
Yr = (10*z['QUARTER'])//10
Qt = (10*z['QUARTER'])%10
z.index = pd.PeriodIndex(year=Yr,quarter=Qt,freq='Q-DEC')
# z.index = pd.period_range('1959Q1', '2003Q4', freq='Q-DEC')
# ex-ante forecast period and assumption
fidx = pd.period_range('2004Q1','2005Q1',freq='Q-DEC')
# constant scenario (0% AGR)
f1 = pd.DataFrame(data=[115.0,115.0,115.0,115.0,115.0],index=fidx)
# pessimistic scenario (-2% AGR)
# f1 = pd.DataFrame(data=[115.0,114.4,113.8,113.2,112.6],index=fidx)
# optimistic scenario (+2% AGR)
# f1 = pd.DataFrame(data=[115.0,115.6,116.2,116.8,117.4],index=fidx)

rgdp = 100*z['GDP']/z['PGDP2000']
growth = 100*(rgdp/rgdp.shift(4)-1.0)
leading = pd.concat([z['LEADING96'],f1])
xvar = pd.concat([leading.shift(1),leading.shift(5)],axis=1)
xvar.columns = ['LEADING-1','LEADING-5']
growth_train = growth['1961Q1':'2001Q4']
xvar_train = xvar['1961Q1':'2001Q4']
growth_test = growth['2002Q1':]
xvar_test = xvar['2002Q1':]

from statsmodels.tsa.arima_model import ARMA
arma1 = ARMA(growth_train,exog=xvar_train,order=(1,4)).fit()
print(arma1.summary())

import matplotlib.pyplot as plt 
%matplotlib inline
for1 = arma1.predict(start='2002Q1',end='2005Q1',exog=xvar_test)
for1.plot()
plt.plot(growth_test)
plt.show()
print(for1)

for2 = arma1.forecast(steps=13,exog=xvar_test,alpha=0.05)
for2[0]  # forecast value
for2[1]  # forecast se
for2[2]  # forecast interval
for2data ={'actural':growth_test, 
           'forecast':for2[0],
           's.e.':for2[1],
           'lower':for2[2][:,0],
           'upper':for2[2][:,1]}
for2index = pd.period_range(start='2002Q1',end='2005Q1',freq='Q-DEC')
forecast = pd.DataFrame(data=for2data,index=for2index)
forecast.plot(y=['actural','forecast','lower','upper'])
print(forecast)
