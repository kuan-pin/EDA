# Example 15.4: GARCH(1,1) Model of DM/BP Exchange Rate
# Bollerslerv and Ghysels [1996], JBES, 307-327.
#
# daily exchange rate data from 1/3/1984 to 12/31/1991 (obs.=1974)
import numpy as np
import pandas as pd
import statsmodels.api as sm

dmbp = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/dmbp.txt",sep='\s+',
                   header=None,names=['xrate','friday'],nrows=1974)

lm2 = sm.OLS.from_formula('xrate~friday',data=dmbp).fit()
lm2.summary()
lm1 = sm.OLS.from_formula('xrate~1',data=dmbp).fit()
lm1.summary()

e = lm1.resid
e2 = e**2

import statsmodels.tsa.api as tsa
import statsmodels.graphics.tsaplots as tsaplt

# time series analysis: identification
tsaplt.plot_acf(e, zero=False, lags=30)
tsaplt.plot_pacf(e, zero=False, lags=30)
tsaplt.plot_acf(e2, zero=False, lags=30)
tsaplt.plot_pacf(e2, zero=False, lags=30)

# https://arch.readthedocs.io/en/latest/univariate/introduction.html
from arch import arch_model

# GARCH(1,1), a common model
# default model: p=1,q=1, normal distribution
garch1 = arch_model(dmbp.xrate).fit()
print(garch1.summary())
garch1.plot()
garch1.arch_lm_test(lags=10,standardized=True)

# Other modelspecifications are possible:
# GJR-GARCH(1,1) model, set o=1
garch2 = arch_model(dmbp.xrate,p=1,o=1,q=1).fit()
print(garch2.summary())
garch2.plot()
# TARCH/ZARCH model, set power=1
garch3 = arch_model(dmbp.xrate,p=1,o=1,q=1,power=1).fit()
print(garch3.summary())
garch3.plot()
# using Stdent's t distribution
garch4 = arch_model(dmbp.xrate,dist='StudentsT').fit()
print(garch4.summary())
garch4.plot()
