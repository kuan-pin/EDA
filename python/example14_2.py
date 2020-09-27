# Example 14.2 Cointegration Test: Engle-Granger Approach 

import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/usyc87.txt",index_col='YEAR',sep='\s+',nrows=66)
y = data['Y']
c = data['C']

from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller

ols1 = sm.OLS.from_formula('C~Y',data=data).fit()
resid1 = ols1.resid    

# test for the unit roots in residuals (null hypothesis)
aeg_test = adfuller(resid1,regression='nc',autolag='AIC',store=True)
aeg_test
print(aeg_test[-1].resols.summary())

def coint_output(res):
    output = pd.Series([res[0],res[1],res[2][0],res[2][1],res[2][2]],
                        index=['Test Statistic','p-value','Critical Value (1%)',
                               'Critical Value (5%)','Critical Value (10%)'])
    print(output) 

# test for no cointegration (null hypothesis)
# default method='aeg', autolag='AIC'
# using the critical values of MacKinnon    
# there is no regression results yet
coint_test1 = coint(c, y, trend='ct', autolag='AIC') 
coint_test1
coint_test2 = coint(c, y, trend='c', autolag='AIC') 
coint_test2
coint_output(coint_test2)
