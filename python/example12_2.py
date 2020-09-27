# Example 12.2 A Nonlinear Rational Expectation Model
# Generalized Method of Moments
# A Nonlinear Rational Expectation Model
# GMM Estimation of Hansen-Singleton Model (Ea, 1982)

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import GMM

data = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/gmmq.txt',sep='\s+',
                  names = ['V1', 'V2', 'V3'])
# data columns: (V1) c(t+1)/c(t) (V2)vwr (V3)rfr
zvar = sm.add_constant(data[1:])  # instrument
xvar = data[:-1] # exog variables
# exog, instrument = map(np.asarray, [x[:-1], z])
yvar = np.zeros(xvar.shape[0]) # endog variable,not used
xvar = np.array(xvar)
zvar = np.array(zvar)

class GMMREM(GMM):

    def momcond(self, params):
        b0, b1 = params
        x = self.exog
        z = self.instrument
        m1 = (z*(b0*(x[:,0]**(b1-1))*x[:,1]-1).reshape(-1,1))
        m2 = (z*(b0*(x[:,0]**(b1-1))*x[:,2]-1).reshape(-1,1))
        return np.column_stack((m1, m2))

# 2 Euler functions with 4 instruments in each equation 
model1 = GMMREM(yvar, xvar, zvar, k_moms=8, k_params=2)
b0 = [1,0]
res1 = model1.fit(b0, maxiter=100, optim_method='bfgs', wargs=dict(centered=False))
print(res1.summary(xname=['beta', 'alpha']))
print(res1.jtest())

# using weights_method='hac' of first-order autocovariance
res2 = model1.fit(b0, maxiter=100, optim_method='bfgs', weights_method='hac',
                  wargs=dict(centered=False,maxlag=1))
print(res2.summary(xname=['beta', 'alpha']))
print(res2.jtest())

# alternatively,
# using NonlinearIVGMM, but one nonlinear fuction only
# from statsmodels.sandbox.regression.gmm import NonlinearIVGMM
