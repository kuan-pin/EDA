# Example 14.1 Testing for Unit Roots
# Augmented Dickey-Fuller Test for Unit Roots 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",index_col='YEAR',sep='\s+',nrows=66)
y = data['Y']
c = data['C']

# select one varible to test
x = c
# first difference of x
# x = c.diff()[1:]

def adf_output(res,reg_result=False):
    r = res[-1]
    output = pd.Series([res[0],res[1],r.usedlag,r.nobs],
                        index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key,value in res[2].items():
        output['Critical Value (%s)' % key] = value
    if reg_result:
        print(r.resols.summary())
    print(output,'\n')   

# dickey-fuller unit-root test
# test for the unit roots of the vaiable (null hypothesis)
# (1) which model? III, II, I (ct, c, nc)
# (2) how many lags augmented?
    
adf_test3 = adfuller(x,regression='ct',autolag='AIC',store=True) # trend and drift
adf_output(adf_test3,reg_result=True)

adf_test2 = adfuller(x,regression='c',autolag='AIC',store=True) # drift
adf_output(adf_test2,reg_result=True)

adf_test1 = adfuller(x,regression='nc',autolag='AIC',store=True) # none
adf_output(adf_test1,reg_result=True)

# in addition, we can test the quardratic trend model (ctt)
adf_test4 = adfuller(x,regression='ctt',autolag='AIC',store=True) # trend and drift
adf_output(adf_test4,reg_result=True)
