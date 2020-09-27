# Example 15.5: GARCH(1,1) Model of DM/BP Exchange Rate
# Bollerslerv and Ghysels [1996], JBES, 307-327.
# Model Variations: GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1)
# Considering Non-Gaussian Distribution: StudentsT, GED
#
# daily exchange rate data from 1/3/1984 to 12/31/1991 (obs.=1974)
import numpy as np
import pandas as pd
import statsmodels.api as sm

dmbp = pd.read_csv("http://web.pdx.edu/~crkl/ceR/data/dmbp.txt",sep='\s+',
                   header=None,names=['xrate','friday'],nrows=1974)

# from arch import arch_model
# model building blocks: mean, variance, distribution
from arch.univariate import ConstantMean,GARCH,EGARCH
from arch.univariate import Normal,StudentsT,GeneralizedError

# GARCH(1,1)
model1 = ConstantMean(dmbp.xrate)
model1.volatility = GARCH(p=1,q=1)
model1.distribution = Normal()
# note: p=#ARCH (lags of squared errors), q=#GARCH (lags of variances)
# how to setup constariants in parametrs?
garch1 = model1.fit()
print(garch1.summary())

# GJR-GARCH(1,1)
model2 = ConstantMean(dmbp.xrate)
model2.volatility = GARCH(p=1,o=1,q=1)
model2.distribution = Normal()
garch2 = model2.fit()
print(garch2.summary())

# Asymmetric EGARCH(1,1) with GED distribution
model3 = ConstantMean(dmbp.xrate)
model3.volatility = EGARCH(p=1,o=1,q=1)
model3.distribution = GeneralizedError()
garch3 = model3.fit()
print(garch3.summary())

# LS + GARCH(1,1)
from arch.univariate import LS
X = np.array(dmbp.friday).reshape(-1,1)
model4 = LS(y=dmbp.xrate,x=X)
model4.volatility = GARCH(p=1,q=1)
model4.distribution = Normal()
garch4 = model4.fit()
print(garch4.summary())

