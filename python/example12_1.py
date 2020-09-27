# Example 12.1 GMM Estimation of a Gamma Distribution
# Generalized Method of Moments
# Estimating Gamma Distribution of Income

import pandas as pd
import numpy as np
from scipy.special import digamma
from statsmodels.sandbox.regression.gmm import GMM

yed = pd.read_csv('http://web.pdx.edu/~crkl/ceR/data/yed20.txt',sep='\s+',nrows=20)
yed = yed.dropna()
print(yed.describe())
y = yed.Income/10 # scale the data may help, y ~ prob. dist.

# https://github.com/josef-pkt/misc/blob/master/notebooks/ex_gmm_gamma.ipynb
# the first method using GMM class in statsmodels
class GMMGamma(GMM):
    
    def __init__(self, *args, **kwds):
        kwds.setdefault('k_moms', 4)
        kwds.setdefault('k_params', 2)
        super(GMMGamma, self).__init__(*args, **kwds)
        
    def momcond(self, params):
        b0, b1 = params
        endog = self.endog
        m1 = endog - b0/b1
        m2 = endog**2 - b0*(b0+1)/(b1**2)
        m3 = np.log(endog) + np.log(b1) - digamma(b0)
        m4 = 1/endog - b1/(b0-1)
        return np.column_stack((m1, m2, m3, m4))

y = np.asarray(y)    
x = np.ones((y.shape[0], 4))  # fake exog
beta0 = np.array([3,1])

model1 = GMMGamma(y, x, None)
# res1 = model1.fit(beta0, maxiter=2, optim_method='nm', wargs=dict(centered=False))
res1 = model1.fit(beta0, maxiter=100, optim_method='bfgs', wargs=dict(centered=False))
print(res1.summary())

# the second method using GMM class in statsmodels
# endog variable only, no exog and instruments given
class GMMGamma2(GMM):
    
    def momcond(self, params):
        b0, b1 = params
        y = self.endog
        m1 = y - b0/b1
        m2 = y**2 - b0*(b0+1)/(b1**2)
        m3 = np.log(y) + np.log(b1) - digamma(b0)
        m4 = 1/y - b1/(b0-1)
        return np.column_stack((m1, m2, m3, m4))

# fake exog and instruments, but k_moms should be given
z = x = np.ones((y.shape[0], 4))

model2 = GMMGamma2(y, x, z)
res2 = model2.fit(beta0, maxiter=100, optim_method='bfgs', wargs=dict(centered=False))
print(res2.summary())

# the third method using GMM class in statsmodels
# fake exog and instruments
x = z = np.ones((y.shape[0], 1))

model3 = GMMGamma2(y, x, z, k_moms=4, k_params=2)
res3 = model3.fit(beta0, maxiter=100, optim_method='bfgs', wargs=dict(centered=False))
print(res3.summary(xname=['alpha', 'beta']))
print(res3.jtest())

# check one-step iteration, weights_method='cov' or HC0
res3a = model3.fit(beta0, maxiter=1, optim_method='bfgs', wargs=dict(centered=False))
print(res3a.summary(xname=['alpha', 'beta']))
# check 2-step iteration
res3b = model3.fit(beta0, maxiter=2, optim_method='bfgs', wargs=dict(centered=False))
print(res3b.summary(xname=['alpha', 'beta']))
# using weights_method='hac' of first-order
res3c = model3.fit(beta0, maxiter=100, optim_method='bfgs', weights_method='hac',
                   wargs=dict(centered=False,maxlag=1))
print(res3c.summary(xname=['alpha', 'beta']))

# check ML case of using 2 moment functions: m1 and m3
# no iteration is needed, but weights_method='cov' is used
class GMMGamma4(GMM):
    
    def momcond(self, params):
        b0, b1 = params
        y = self.endog
        m1 = y - b0/b1
        m2 = y**2 - b0*(b0+1)/(b1**2)
        m3 = np.log(y) + np.log(b1) - digamma(b0)
        m4 = 1/y - b1/(b0-1)
        return np.column_stack((m1, m3))
    
model4 = GMMGamma4(y, x, z, k_moms=2, k_params=2)
res4 = model4.fit(beta0, maxiter=100, optim_method='bfgs', wargs=dict(centered=False))
print(res4.summary(xname=['alpha', 'beta']))
